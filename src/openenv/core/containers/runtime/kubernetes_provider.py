# SPDX-License-Identifier: BSD-3-Clause

"""
Kubernetes container provider for running OpenEnv environments in-cluster.

Requires the ``kubernetes`` client: ``pip install openenv[kubernetes]``

One provider instance owns one Pod and one Service, matching
``LocalDockerProvider`` owning one container. The returned ``base_url`` is the
short in-cluster Service address ``http://{name}.{namespace}:{port}``. Pod DNS
search completes that to the cluster domain, so the provider does not hardcode
``.svc.cluster.local``. That plaintext URL is the in-cluster exception
confirmed for this provider: ``ModalProvider`` and ``ACASandboxProvider`` still
refuse non-HTTPS tunnel URLs. This provider never returns a URL it did not
construct from a validated name, namespace, and port, and it does not copy
that URL or raw API bodies into errors.
"""

from __future__ import annotations

import re
import secrets
import time
from typing import Any, Mapping, Optional

from .providers import ContainerProvider

_CONTAINER_PORT = 8000
_CONTAINER_NAME = "env"
_DEFAULT_NAMESPACE = "default"
_NAME_PREFIX = "openenv-"

# ``openenv-`` + slug + ``-`` + millisecond timestamp + ``-`` + 4 hex chars.
_SLUG_MAX = 32

_DNS_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_LABEL_NAME = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9_.-]{0,61}[A-Za-z0-9])?$")
_LABEL_VALUE = re.compile(r"^([A-Za-z0-9]([A-Za-z0-9_.-]{0,61}[A-Za-z0-9])?)?$")
_SAFE_REASON = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,63}$")

# Applied last so a caller-supplied label cannot drop ownership or retarget
# the Service selector.
_SELECTOR_KEY = "app.kubernetes.io/instance"

_START_KWARGS = {
    "resources",
    "runtime_class_name",
    "image_pull_secrets",
    "labels",
    "service_account_name",
    "automount_service_account_token",
}


def _raise_install_error(exc: ImportError) -> None:
    raise RuntimeError(
        "KubernetesProvider requires the kubernetes client. "
        "Install it with `pip install openenv[kubernetes]`."
    ) from exc


def _dns_label(value: str, *, max_length: int, what: str) -> str:
    """Return *value* as a DNS-1123 label, or raise ``ValueError``.

    Namespace and runtime class names are validated, not rewritten: rewriting
    them would target a different object than the one the caller named.
    """
    if not isinstance(value, str) or not _DNS_LABEL.fullmatch(value):
        raise ValueError(
            f"{what} must be a DNS-1123 label: lowercase alphanumeric "
            "characters or '-', starting and ending with an alphanumeric."
        )
    if not 1 <= len(value) <= max_length:
        raise ValueError(f"{what} must be 1 to {max_length} characters.")
    return value


def _slug(image: str) -> str:
    """DNS-1123 slug from an image reference. Raises if nothing usable remains."""
    raw = image.split("/")[-1].split("@")[0].split(":")[0]
    cleaned = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    cleaned = re.sub(r"-{2,}", "-", cleaned)[:_SLUG_MAX].strip("-")
    if not cleaned or not _DNS_LABEL.fullmatch(cleaned):
        raise ValueError(
            "image does not yield a valid Kubernetes name. "
            "Use an image reference with an alphanumeric name."
        )
    return cleaned


def _resource_name(image: str) -> str:
    suffix = f"{int(time.time() * 1000)}-{secrets.token_hex(2)}"
    name = f"{_NAME_PREFIX}{_slug(image)}-{suffix}"
    if len(name) > 63 or not _DNS_LABEL.fullmatch(name):
        raise ValueError("generated Kubernetes name is not a valid DNS-1123 label.")
    return name


def _label_key(key: str) -> str:
    if not isinstance(key, str) or len(key) > 253:
        raise ValueError(
            "Kubernetes label keys must be strings of at most 253 characters."
        )
    prefix, sep, name = key.rpartition("/")
    if not _LABEL_NAME.fullmatch(name):
        raise ValueError(f"Invalid Kubernetes label key {key!r}.")
    if sep and (len(prefix) > 253 or not _dns_subdomain(prefix)):
        raise ValueError(f"Invalid Kubernetes label key prefix {prefix!r}.")
    return key


def _dns_subdomain(value: str) -> bool:
    if not value or len(value) > 253:
        return False
    return all(_DNS_LABEL.fullmatch(part) for part in value.split("."))


def _label_value(value: str) -> str:
    if not isinstance(value, str) or not _LABEL_VALUE.fullmatch(value):
        raise ValueError(
            "Kubernetes label values must be 63 characters or fewer and contain "
            "only alphanumeric characters, '-', '_' or '.'."
        )
    return value


def _ownership_labels(name: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "openenv",
        "app.kubernetes.io/instance": name,
        "app.kubernetes.io/managed-by": "openenv",
        "app.kubernetes.io/part-of": "openenv",
        "app.kubernetes.io/component": "environment",
    }


def _merged_labels(name: str, extra: Optional[Mapping[str, str]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    if extra:
        if not isinstance(extra, Mapping):
            raise ValueError("labels must be a mapping of strings.")
        for key, value in extra.items():
            labels[_label_key(key)] = _label_value(value)
    labels.update(_ownership_labels(name))
    return labels


def _safe_reason(reason: Optional[str]) -> Optional[str]:
    """Keep kubelet reason codes. Drop anything else so API text cannot leak."""
    if isinstance(reason, str) and _SAFE_REASON.fullmatch(reason):
        return reason
    return None


class _NamespaceMissing(RuntimeError):
    """Creating a namespaced object returned 404: the namespace is absent."""


class _CreateConflict(RuntimeError):
    """Creating a namespaced object returned 409: the name is already taken."""


class _TransientStatusError(RuntimeError):
    """A Pod status read can be retried within the readiness deadline."""


def _incluster_base_url(name: str, namespace: str, port: int) -> str:
    # *name* and *namespace* are already DNS-1123 labels and *port* is an int,
    # so this cannot be steered by a status field from the API server.
    # ``name.namespace`` is completed by the Pod DNS search list
    # (``svc.<cluster-domain>``), which works when the cluster domain is not
    # ``cluster.local``.
    return f"http://{name}.{namespace}:{port}"


def _owned_by_provider(labels: Any, name: str) -> bool:
    if not isinstance(labels, Mapping):
        return False
    return (
        labels.get("app.kubernetes.io/managed-by") == "openenv"
        and labels.get("app.kubernetes.io/instance") == name
    )


class _DefaultKubernetesAdapter:
    """Thin adapter over the official ``kubernetes`` client.

    The provider talks to this private adapter instead of spreading SDK details
    through its own logic; tests inject a duck-typed fake in its place.
    """

    def __init__(
        self,
        *,
        namespace: str,
        kubeconfig: Optional[str],
        context: Optional[str],
    ):
        try:
            from kubernetes import client, config
            from kubernetes.client.exceptions import ApiException
        except ImportError as exc:
            _raise_install_error(exc)

        self._client = client
        self._ApiException = ApiException
        self._namespace = namespace
        self._load_config(config, kubeconfig=kubeconfig, context=context)
        configuration = client.Configuration.get_default_copy()
        # Readiness owns the retry deadline; SDK retries would multiply it.
        configuration.retries = 0
        self._api = client.CoreV1Api(client.ApiClient(configuration=configuration))

    def _load_config(
        self, config: Any, *, kubeconfig: Optional[str], context: Optional[str]
    ) -> None:
        if kubeconfig is not None or context is not None:
            config.load_kube_config(config_file=kubeconfig, context=context)
            return
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

    def _api_error(self, exc: Exception, action: str, *, missing_ok: bool) -> None:
        status = getattr(exc, "status", None)
        if missing_ok and status == 404:
            return
        if status == 409:
            raise _CreateConflict(
                f"Kubernetes {action} name is already in use"
            ) from None
        if status == 404 and action.startswith("create "):
            raise _NamespaceMissing(
                "Kubernetes namespace does not exist. "
                "KubernetesProvider does not create namespaces."
            ) from None
        # The API body can echo object contents. Do not chain it into the error.
        raise RuntimeError(f"Failed to {action} Kubernetes resource") from None

    def create_pod(self, manifest: Mapping[str, Any]) -> None:
        try:
            self._api.create_namespaced_pod(self._namespace, self._pod(manifest))
        except self._ApiException as exc:
            self._api_error(exc, "create Pod", missing_ok=False)

    def create_service(self, manifest: Mapping[str, Any]) -> None:
        try:
            self._api.create_namespaced_service(
                self._namespace, self._service(manifest)
            )
        except self._ApiException as exc:
            self._api_error(exc, "create Service", missing_ok=False)

    def pod_labels(self, name: str) -> Optional[dict[str, str]]:
        """Return Pod labels, or ``None`` when the Pod does not exist."""
        return self._resource_labels(name, self._api.read_namespaced_pod, "read Pod")

    def service_labels(self, name: str) -> Optional[dict[str, str]]:
        """Return Service labels, or ``None`` when the Service does not exist."""
        return self._resource_labels(
            name, self._api.read_namespaced_service, "read Service"
        )

    def _resource_labels(
        self, name: str, read: Any, action: str
    ) -> Optional[dict[str, str]]:
        try:
            obj = read(name, self._namespace)
        except self._ApiException as exc:
            if getattr(exc, "status", None) == 404:
                return None
            self._api_error(exc, action, missing_ok=False)
            raise RuntimeError(f"Failed to {action} Kubernetes resource") from None
        metadata = getattr(obj, "metadata", None)
        labels = getattr(metadata, "labels", None) if metadata is not None else None
        if not isinstance(labels, Mapping):
            return {}
        return {str(key): str(value) for key, value in labels.items()}

    def delete_pod(self, name: str) -> None:
        try:
            self._api.delete_namespaced_pod(name, self._namespace)
        except self._ApiException as exc:
            self._api_error(exc, "delete Pod", missing_ok=True)

    def delete_service(self, name: str) -> None:
        try:
            self._api.delete_namespaced_service(name, self._namespace)
        except self._ApiException as exc:
            self._api_error(exc, "delete Service", missing_ok=True)

    def pod_status(
        self, name: str, timeout_s: float = 30.0
    ) -> dict[str, Optional[str]]:
        from urllib3.exceptions import HTTPError

        try:
            # Older supported SDKs ignore a float timeout. Split the budget
            # between connect and read using their supported tuple form.
            pod = self._api.read_namespaced_pod(
                name,
                self._namespace,
                _request_timeout=(timeout_s / 2, timeout_s / 2),
            )
        except Exception as exc:
            status = getattr(exc, "status", None)
            if (
                isinstance(exc, (TimeoutError, HTTPError))
                or status == 429
                or (isinstance(status, int) and 500 <= status < 600)
            ):
                raise _TransientStatusError(
                    "Kubernetes Pod status read failed"
                ) from None
            self._api_error(exc, "read Pod", missing_ok=False)
            raise RuntimeError("Failed to read Kubernetes Pod") from None
        status = getattr(pod, "status", None)
        phase = getattr(status, "phase", None) if status is not None else None
        reason = None
        if status is not None:
            for container in status.container_statuses or []:
                state = getattr(container, "state", None)
                waiting = getattr(state, "waiting", None) if state is not None else None
                terminated = (
                    getattr(state, "terminated", None) if state is not None else None
                )
                if waiting is not None and getattr(waiting, "reason", None):
                    reason = waiting.reason
                elif terminated is not None and getattr(terminated, "reason", None):
                    reason = terminated.reason
        return {"phase": phase or "Unknown", "reason": reason}

    def _metadata(self, meta: Mapping[str, Any]) -> Any:
        return self._client.V1ObjectMeta(
            name=meta["name"],
            namespace=meta["namespace"],
            labels=dict(meta["labels"]),
        )

    def _pod(self, manifest: Mapping[str, Any]) -> Any:
        spec = manifest["spec"]
        container = spec["containers"][0]
        container_kwargs: dict[str, Any] = {
            "name": container["name"],
            "image": container["image"],
            "ports": [
                self._client.V1ContainerPort(
                    name=port["name"], container_port=port["container_port"]
                )
                for port in container["ports"]
            ],
        }
        if container.get("env"):
            container_kwargs["env"] = [
                self._client.V1EnvVar(name=item["name"], value=item["value"])
                for item in container["env"]
            ]
        resources = container.get("resources")
        if resources:
            container_kwargs["resources"] = self._client.V1ResourceRequirements(
                requests=resources.get("requests"),
                limits=resources.get("limits"),
            )
        pod_kwargs: dict[str, Any] = {
            "restart_policy": spec["restart_policy"],
            "automount_service_account_token": spec["automount_service_account_token"],
            "containers": [self._client.V1Container(**container_kwargs)],
        }
        if spec.get("runtime_class_name"):
            pod_kwargs["runtime_class_name"] = spec["runtime_class_name"]
        if spec.get("service_account_name"):
            pod_kwargs["service_account_name"] = spec["service_account_name"]
        if spec.get("image_pull_secrets"):
            pod_kwargs["image_pull_secrets"] = [
                self._client.V1LocalObjectReference(name=secret)
                for secret in spec["image_pull_secrets"]
            ]
        return self._client.V1Pod(
            metadata=self._metadata(manifest["metadata"]),
            spec=self._client.V1PodSpec(**pod_kwargs),
        )

    def _service(self, manifest: Mapping[str, Any]) -> Any:
        spec = manifest["spec"]
        port = spec["ports"][0]
        return self._client.V1Service(
            metadata=self._metadata(manifest["metadata"]),
            spec=self._client.V1ServiceSpec(
                selector=dict(spec["selector"]),
                ports=[
                    self._client.V1ServicePort(
                        name=port["name"],
                        port=port["port"],
                        target_port=port["target_port"],
                        protocol=port["protocol"],
                    )
                ],
            ),
        )


class KubernetesProvider(ContainerProvider):
    """
    Container provider that runs one environment as one Pod and one Service.

    ``start_container`` returns ``http://{name}.{namespace}:{port}``. That
    address is reachable only from inside the cluster: Pod DNS search expands
    it under ``svc.<cluster-domain>``. It is plaintext on purpose: in-cluster
    HTTP is the connectivity decision for this provider. Cloud sandbox
    providers keep their own HTTPS requirement.

    The namespace must already exist. This provider does not create namespaces,
    Deployments, Ingress objects, or port-forwards.

    Only one Pod is active per provider. Calling ``start_container`` again
    before ``stop_container()`` / ``close()`` raises ``RuntimeError`` rather
    than orphaning the running Pod. A failed start deletes a resource only
    after its OpenEnv ownership labels match. When a create response is lost
    and that read fails, the resource stays unconfirmed: ``stop_container``
    reads the labels again and deletes only on a match. ``stop_container`` is
    safe to call more than once.

    The Pod does not mount a service-account token unless
    ``automount_service_account_token=True`` is passed to ``start_container``.

    Example:
        ```python
        from openenv.core.containers.runtime.kubernetes_provider import (
            KubernetesProvider,
        )

        provider = KubernetesProvider(namespace="openenv")
        base_url = provider.start_container("echo-env:latest")
        provider.wait_for_ready(base_url)
        provider.stop_container()
        ```
    """

    def __init__(
        self,
        *,
        image: str | None = None,
        env_vars: dict[str, str] | None = None,
        namespace: str = _DEFAULT_NAMESPACE,
        kubeconfig: str | None = None,
        context: str | None = None,
        _adapter: Any = None,
    ):
        """
        Args:
            image (`str`, *optional*):
                Registry image used when ``start_container()`` is called
                without an image.
            env_vars (`dict`, *optional*):
                Environment variables used when ``start_container()`` is called
                without ``env_vars``.
            namespace (`str`, *optional*, defaults to ``"default"``):
                Existing namespace for the Pod and Service. Not created.
            kubeconfig (`str`, *optional*):
                Kubeconfig path. When omitted, in-cluster config is tried and
                then the default kubeconfig.
            context (`str`, *optional*):
                Kubeconfig context. Implies kubeconfig loading rather than
                in-cluster config.
        """
        self._image = image
        self._env_vars = env_vars
        self._namespace = _dns_label(namespace, max_length=63, what="namespace")
        self._name: str | None = None
        self._base_url: str | None = None
        self._service_port: int | None = None
        self._pod_created = False
        self._service_created = False
        self._pod_unconfirmed = False
        self._service_unconfirmed = False

        if _adapter is None:
            self._adapter: Any = _DefaultKubernetesAdapter(
                namespace=self._namespace,
                kubeconfig=kubeconfig,
                context=context,
            )
        else:
            self._adapter = _adapter

    @property
    def base_url(self) -> str:
        """URL returned by the last successful ``start_container``."""
        if self._base_url is None:
            raise RuntimeError(
                "KubernetesProvider has no active base_url. Start the provider "
                "before reading base_url."
            )
        return self._base_url

    def start_container(
        self,
        image: str | None = None,
        port: int | None = None,
        env_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Create one Pod and one Service and return the in-cluster base URL.

        Args:
            image (`str`, *optional*):
                Registry image. May be omitted when supplied to the constructor.
            port (`int`, *optional*):
                Service port. Defaults to 8000. The container always listens
                on 8000, matching ``LocalDockerProvider``.
            env_vars (`dict`, *optional*):
                Environment variables for the container. Replaces constructor
                ``env_vars`` when passed, including an empty dict.
            **kwargs:
                ``resources``, ``runtime_class_name``, ``image_pull_secrets``,
                ``labels``, ``service_account_name``, and
                ``automount_service_account_token``. Unknown keys raise
                ``ValueError``.

        Returns:
            `str`: ``http://{name}.{namespace}:{port}``.
        """
        if self._has_tracked_resources() or self._base_url is not None:
            raise RuntimeError(
                "KubernetesProvider already has an active Pod. Call "
                "stop_container() (or close()) before starting another — a "
                "second start would orphan the running Pod."
            )

        unknown = set(kwargs) - _START_KWARGS
        if unknown:
            raise ValueError(
                f"Unsupported kwargs for KubernetesProvider: {sorted(unknown)}"
            )

        effective_image = image if image is not None else self._image
        if not isinstance(effective_image, str) or not effective_image:
            raise ValueError(
                "KubernetesProvider requires an image. Pass it to the "
                "constructor or start_container()."
            )
        service_port = _CONTAINER_PORT if port is None else port
        if isinstance(service_port, bool) or not isinstance(service_port, int):
            raise ValueError("port must be an integer.")
        if not 1 <= service_port <= 65535:
            raise ValueError("port must be between 1 and 65535.")

        effective_env = self._env_vars if env_vars is None else env_vars
        env_list = _env_list(effective_env)
        resources = _resources(kwargs.get("resources"))
        runtime_class_name = _optional_dns_subdomain(
            kwargs.get("runtime_class_name"), what="runtime_class_name"
        )
        service_account_name = _optional_dns_subdomain(
            kwargs.get("service_account_name"), what="service_account_name"
        )
        image_pull_secrets = _secret_names(kwargs.get("image_pull_secrets"))
        automount = kwargs.get("automount_service_account_token", False)
        if not isinstance(automount, bool):
            raise ValueError("automount_service_account_token must be a bool.")

        name = _resource_name(effective_image)
        labels = _merged_labels(name, kwargs.get("labels"))
        self._name = name
        self._service_port = service_port
        try:
            self._adapter.create_pod(
                _pod_manifest(
                    name=name,
                    namespace=self._namespace,
                    labels=labels,
                    image=effective_image,
                    env_list=env_list,
                    resources=resources,
                    runtime_class_name=runtime_class_name,
                    service_account_name=service_account_name,
                    image_pull_secrets=image_pull_secrets,
                    automount_service_account_token=automount,
                )
            )
            self._pod_created = True
            self._adapter.create_service(
                _service_manifest(
                    name=name,
                    namespace=self._namespace,
                    labels=labels,
                    service_port=service_port,
                )
            )
            self._service_created = True
        except BaseException as exc:
            # Interrupts can also arrive after the API created a resource.
            # A cleanup failure must not mask the original error.
            try:
                self._cleanup_after_failed_start(exc)
            except Exception:
                pass
            raise

        self._base_url = _incluster_base_url(name, self._namespace, service_port)
        return self._base_url

    def _cleanup_after_failed_start(self, exc: BaseException) -> None:
        """Delete objects this start may have created.

        A 409 is someone else's name, and a 404 means the namespace is
        missing, so those do not probe. Any other create error may mean the
        API server applied the object and the client lost the response. Only
        the resource whose create was attempted is checked. It is deleted only
        when the OpenEnv ownership labels match. A failed read is recorded as
        unconfirmed, which is not the same as a successful create: a later
        ``stop_container()`` reads the labels again before deleting.
        """
        uncertain = not isinstance(exc, (_CreateConflict, _NamespaceMissing))
        # Service create runs only after Pod create has succeeded.
        service_was_attempted = self._pod_created
        self._release("pod", uncertain=uncertain)
        self._release("service", uncertain=uncertain and service_was_attempted)
        if not self._has_tracked_resources():
            self._name = None
            self._base_url = None
            self._service_port = None

    def _release(self, kind: str, *, uncertain: bool) -> None:
        if self._created(kind):
            self._delete_tracked(kind)
            return
        if not uncertain:
            return
        owned = self._lookup_owned(kind)
        if owned is None:
            self._set_unconfirmed(kind, True)
            return
        if not owned:
            return
        self._set_created(kind, True)
        self._delete_tracked(kind)

    def _lookup_owned(self, kind: str) -> Optional[bool]:
        """Return whether *kind* at ``_name`` is ours.

        ``None`` means the read itself failed. Callers record that as
        unconfirmed. They do not treat it as a confirmed create.
        """
        if self._name is None:
            return False
        try:
            if kind == "pod":
                labels = self._adapter.pod_labels(self._name)
            else:
                labels = self._adapter.service_labels(self._name)
        except Exception:
            return None
        if labels is None:
            return False
        return _owned_by_provider(labels, self._name)

    def _delete_tracked(self, kind: str) -> None:
        """Delete a resource already known to be ours.

        A failed delete stays confirmed so ``stop_container()`` can retry it.
        """
        if self._name is None:
            return
        try:
            if kind == "pod":
                self._adapter.delete_pod(self._name)
            else:
                self._adapter.delete_service(self._name)
        except Exception:
            self._set_created(kind, True)
            return
        self._set_created(kind, False)

    def _created(self, kind: str) -> bool:
        if kind == "pod":
            return self._pod_created
        return self._service_created

    def _set_created(self, kind: str, created: bool) -> None:
        if kind == "pod":
            self._pod_created = created
        else:
            self._service_created = created

    def _unconfirmed(self, kind: str) -> bool:
        if kind == "pod":
            return self._pod_unconfirmed
        return self._service_unconfirmed

    def _set_unconfirmed(self, kind: str, unconfirmed: bool) -> None:
        if kind == "pod":
            self._pod_unconfirmed = unconfirmed
        else:
            self._service_unconfirmed = unconfirmed

    def _has_tracked_resources(self) -> bool:
        return (
            self._pod_created
            or self._service_created
            or self._pod_unconfirmed
            or self._service_unconfirmed
        )

    def stop_container(self) -> None:
        """Delete the Pod and Service this provider created.

        Safe when nothing was started and safe to call twice. A 404 from the
        API is treated as already deleted. A confirmed resource is deleted by
        name. An unconfirmed one is deleted only after a fresh ownership-label
        match. If that read fails, the unconfirmed state is kept for another
        retry. Objects this instance does not own are left in place, including
        a name that was already taken.
        """
        if self._name is None and not self._has_tracked_resources():
            self._base_url = None
            self._service_port = None
            return

        errors: list[str] = []
        self._stop_kind("pod", errors)
        self._stop_kind("service", errors)
        if errors:
            raise RuntimeError(
                "Failed to delete Kubernetes resources created by this provider. "
                "Call stop_container() again to retry."
            )
        self._name = None
        self._base_url = None
        self._service_port = None

    def _stop_kind(self, kind: str, errors: list[str]) -> None:
        if self._name is None:
            return
        if self._unconfirmed(kind):
            owned = self._lookup_owned(kind)
            if owned is None:
                errors.append(kind)
                return
            self._set_unconfirmed(kind, False)
            if not owned:
                return
            self._set_created(kind, True)
        if not self._created(kind):
            return
        try:
            if kind == "pod":
                self._adapter.delete_pod(self._name)
            else:
                self._adapter.delete_service(self._name)
        except Exception:
            errors.append(kind)
            return
        self._set_created(kind, False)

    def close(self) -> None:
        """Stop the active Pod and Service.

        Overrides the base no-op so a caller holding a bare ``ContainerProvider``
        reference can release the Pod polymorphically (also invoked on
        context-manager exit). Equivalent to ``stop_container()``.
        """
        self.stop_container()

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

    def wait_for_ready(self, base_url: str, timeout_s: float = 120.0) -> None:
        """
        Wait until the Pod is Running and ``GET /health`` returns 200.

        A Failed or Succeeded Pod raises immediately instead of consuming the
        timeout. The URL is not included in errors: this method only polls the
        Service URL returned by ``start_container``.

        Args:
            base_url (`str`):
                URL returned by ``start_container()``.
            timeout_s (`float`, *optional*, defaults to `120.0`):
                Maximum seconds to wait. Image pulls are slower than local Docker.

        Raises:
            TimeoutError: If the Pod does not become ready in time.
            RuntimeError: If the Pod has already ended.
            ValueError: If ``base_url`` is not the URL this provider returned.
        """
        import requests

        expected = self._base_url
        if expected is None or not isinstance(base_url, str):
            raise ValueError(
                "base_url does not match the in-cluster Service URL for this provider."
            )
        if base_url.rstrip("/") != expected:
            raise ValueError(
                "base_url does not match the in-cluster Service URL for this provider."
            )
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ValueError("timeout_s must be a number.")
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative.")

        health_url = f"{expected}/health"
        proxies = {"http": "", "https": "", "all": ""}
        deadline = time.monotonic() + timeout_s
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                phase, reason = self._phase(remaining)
            except _TransientStatusError:
                phase, reason = "Unknown", None
            if phase in ("Failed", "Succeeded"):
                raise RuntimeError(self._ended_message(phase, reason))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if phase == "Running":
                try:
                    response = requests.get(
                        health_url, timeout=min(2.0, remaining), proxies=proxies
                    )
                    if response.status_code == 200:
                        return
                except requests.RequestException:
                    pass
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))

        raise TimeoutError(f"Kubernetes Pod did not become ready within {timeout_s}s.")

    def _phase(self, timeout_s: float) -> tuple[str, Optional[str]]:
        if self._name is None:
            raise RuntimeError(
                "KubernetesProvider has no active Pod. Start the provider "
                "before waiting for readiness."
            )
        status = self._adapter.pod_status(self._name, timeout_s=timeout_s)
        phase = status.get("phase") if isinstance(status, Mapping) else None
        reason = status.get("reason") if isinstance(status, Mapping) else None
        if not isinstance(phase, str) or not _SAFE_REASON.fullmatch(phase):
            phase = "Unknown"
        return phase, _safe_reason(reason if isinstance(reason, str) else None)

    def _ended_message(self, phase: str, reason: Optional[str]) -> str:
        if reason is None:
            return (
                f"Kubernetes Pod entered phase {phase} before it became ready. "
                "The Service URL and Pod status message are omitted."
            )
        return (
            f"Kubernetes Pod entered phase {phase} ({reason}) before it became "
            "ready. The Service URL and Pod status message are omitted."
        )


def _env_list(env_vars: Optional[Mapping[str, str]]) -> list[dict[str, str]]:
    if env_vars is None:
        return []
    if not isinstance(env_vars, Mapping):
        raise ValueError("env_vars must be a mapping of strings.")
    items: list[dict[str, str]] = []
    for key, value in env_vars.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("env_vars keys and values must be strings.")
        items.append({"name": key, "value": value})
    return items


def _resources(resources: Any) -> Optional[dict[str, dict[str, str]]]:
    if resources is None:
        return None
    if not isinstance(resources, Mapping):
        raise ValueError("resources must be a mapping of requests and limits.")
    unknown = set(resources) - {"requests", "limits"}
    if unknown:
        raise ValueError(
            f"Unsupported resources keys for KubernetesProvider: {sorted(unknown)}"
        )
    parsed: dict[str, dict[str, str]] = {}
    for section in ("requests", "limits"):
        values = resources.get(section)
        if values is None:
            continue
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"resources.{section} must be a non-empty mapping.")
        parsed[section] = {}
        for key, value in values.items():
            if not isinstance(key, str) or not isinstance(value, str) or not value:
                raise ValueError(
                    f"resources.{section} keys and values must be non-empty strings."
                )
            parsed[section][key] = value
    return parsed or None


def _optional_dns_subdomain(value: Any, *, what: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not _dns_subdomain(value):
        raise ValueError(
            f"{what} must be a DNS-1123 subdomain: lowercase labels separated by '.'."
        )
    return value


def _secret_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError("image_pull_secrets must be a list of secret names.")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or not _dns_subdomain(item):
            raise ValueError("image_pull_secrets entries must be DNS-1123 subdomains.")
        names.append(item)
    return names


def _pod_manifest(
    *,
    name: str,
    namespace: str,
    labels: dict[str, str],
    image: str,
    env_list: list[dict[str, str]],
    resources: Optional[dict[str, dict[str, str]]],
    runtime_class_name: Optional[str],
    service_account_name: Optional[str],
    image_pull_secrets: list[str],
    automount_service_account_token: bool,
) -> dict[str, Any]:
    container: dict[str, Any] = {
        "name": _CONTAINER_NAME,
        "image": image,
        "ports": [{"name": "http", "container_port": _CONTAINER_PORT}],
    }
    if env_list:
        container["env"] = env_list
    if resources:
        container["resources"] = resources
    spec: dict[str, Any] = {
        "restart_policy": "Never",
        "automount_service_account_token": automount_service_account_token,
        "containers": [container],
    }
    if runtime_class_name:
        spec["runtime_class_name"] = runtime_class_name
    if service_account_name:
        spec["service_account_name"] = service_account_name
    if image_pull_secrets:
        spec["image_pull_secrets"] = image_pull_secrets
    return {
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": spec,
    }


def _service_manifest(
    *,
    name: str,
    namespace: str,
    labels: dict[str, str],
    service_port: int,
) -> dict[str, Any]:
    return {
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {
            "selector": {_SELECTOR_KEY: name},
            "ports": [
                {
                    "name": "http",
                    "port": service_port,
                    "target_port": _CONTAINER_PORT,
                    "protocol": "TCP",
                }
            ],
        },
    }


__all__ = ["KubernetesProvider"]
