# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Pelican SVG environment.

Nothing here touches the network. The vision judge is exercised through a stub
client, which is also the only way to test the failure paths that matter, such
as a judge that raises and must not end up inflating the reward.
"""

from __future__ import annotations

import asyncio
import importlib
import math
import pathlib
import sys
import threading

import pytest

pytest.importorskip("resvg_py", reason="resvg-py is not installed")

from envs.pelican_svg_env.models import PelicanSvgAction
from envs.pelican_svg_env.server.gate import run_gate
from envs.pelican_svg_env.server.geometry import (
    apply,
    extract_shapes,
    flatten_path,
    length,
    parse_transform,
    significant_shapes,
)
from envs.pelican_svg_env.server.pelican_svg_environment import (
    _judge_from_env,
    PelicanSvgEnvironment,
)
from envs.pelican_svg_env.server.render import image_stats, render_png, RenderError
from envs.pelican_svg_env.server.rubric import build_rubric
from envs.pelican_svg_env.server.scoring import (
    component_weights,
    evaluate_deterministic,
    evaluate_submission,
)
from envs.pelican_svg_env.server.structure import analyse_structure, find_wheels
from envs.pelican_svg_env.server.svg_source import (
    extract_svg,
    inspect_source,
    parse_svg,
    SvgParseError,
    TruncatedSvgError,
)
from envs.pelican_svg_env.server.tasks import (
    all_tasks,
    make_task,
    sample_task,
    task_from_ids,
)
from envs.pelican_svg_env.server.vision_judge import (
    DEFAULT_JUDGE_MODEL,
    HFVisionClient,
    VisionJudge,
)

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "envs/pelican_svg_env/fixtures"

# Fixtures that are honest drawing attempts and must reach the scoring layers.
ADMISSIBLE = {
    "good_pelican_bike",
    "wheels_as_paths",
    "wheels_via_use",
    "medium_pelican_bike",
    "bike_no_bird",
    "bird_no_bike",
    "bad_scribble",
}

PELICAN_TERMS = ["pelican", "bicycle", "bike", "bird"]


def fixture(name: str) -> str:
    return (FIXTURES / f"{name}.svg").read_text()


def svg(body: str, view_box: str = "0 0 100 100") -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}">{body}</svg>'


class TestExtraction:
    """Pulling the SVG out of a raw model reply."""

    def test_extracts_from_fenced_block(self):
        reply = "Sure!\n```svg\n<svg><circle r='1'/></svg>\n```\nHope that helps."
        assert extract_svg(reply) == "<svg><circle r='1'/></svg>"

    def test_extracts_from_surrounding_prose(self):
        assert extract_svg("Here: <svg><rect/></svg> done!") == "<svg><rect/></svg>"

    def test_last_attempt_wins(self):
        reply = "First:\n<svg><rect id='a'/></svg>\nBetter:\n<svg><rect id='b'/></svg>"
        assert "id='b'" in extract_svg(reply)

    def test_missing_svg_raises(self):
        with pytest.raises(SvgParseError):
            extract_svg("I cannot draw that.")

    def test_truncated_svg_is_distinguished_from_absent(self):
        """A generation cut off by a token limit is a different fact."""
        with pytest.raises(TruncatedSvgError):
            extract_svg("<svg xmlns='http://www.w3.org/2000/svg'><circle r='1'/>")


class TestSourceGate:
    """Source-level rejection of degenerate and bad-faith submissions."""

    def test_clean_drawing_has_no_violations(self):
        report = inspect_source(fixture("good_pelican_bike"), PELICAN_TERMS)
        assert report.ok, report.codes

    def test_rejects_data_uri_raster(self):
        """The cheat that renders convincingly and only shows up in the source."""
        report = inspect_source(fixture("cheat_raster_datauri"), PELICAN_TERMS)
        assert "embedded_raster" in report.codes

    def test_rejects_xlink_href_raster(self):
        body = (
            '<image xlink:href="data:image/png;base64,AAAA" width="10" height="10"/>'
            '<circle cx="1" cy="1" r="1"/><rect width="2" height="2"/>'
        )
        document = (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink">{body}</svg>'
        )
        assert "embedded_raster" in inspect_source(document).codes

    def test_rejects_external_reference(self):
        body = '<use href="https://evil.example/x.svg"/><circle r="1"/><rect/><line/>'
        assert "external_reference" in inspect_source(svg(body)).codes

    def test_rejects_script_element(self):
        body = '<script>fetch("/")</script><circle r="1"/><rect/><line/>'
        assert "forbidden_element" in inspect_source(svg(body)).codes

    def test_rejects_text_naming_the_subject(self):
        body = '<text>a pelican</text><circle r="1"/><rect/><line/>'
        assert "text_label" in inspect_source(svg(body), PELICAN_TERMS).codes

    def test_rejects_subject_named_inside_tspan(self):
        body = '<text><tspan>a</tspan> pelican</text><circle r="1"/><rect/><line/>'
        assert "text_label" in inspect_source(svg(body), PELICAN_TERMS).codes

    def test_text_after_closing_tag_is_not_rendered_and_not_counted(self):
        """An element's tail sits outside it, so it never reaches the canvas."""
        body = '<text>hi</text>pelican<circle r="1"/><rect/><line/>'
        assert "text_label" not in inspect_source(svg(body), PELICAN_TERMS).codes

    def test_rejects_entity_declarations(self):
        """ElementTree expands internal entities, so a DTD is a billion-laughs vector."""
        bomb = (
            '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">]>'
            '<svg xmlns="http://www.w3.org/2000/svg"><text>&lol;</text></svg>'
        )
        with pytest.raises(SvgParseError, match="DTD"):
            parse_svg(bomb)

    def test_rejects_oversized_source(self):
        with pytest.raises(SvgParseError, match="exceeds"):
            parse_svg(svg("<rect/>" * 40000))

    def test_rejects_non_svg_root(self):
        with pytest.raises(SvgParseError, match="root element"):
            parse_svg(
                '<html><body><svg xmlns="http://www.w3.org/2000/svg"/></body></html>'
            )


class TestRender:
    """Rasterisation and the measurements taken from it."""

    def test_render_is_deterministic(self):
        source = fixture("good_pelican_bike")
        digests = {render_png(source, 256) for _ in range(3)}
        assert len(digests) == 1

    def test_malformed_source_raises(self):
        with pytest.raises(RenderError):
            render_png(fixture("malformed"))

    def test_blank_canvas_has_no_ink(self):
        assert image_stats(render_png(fixture("blank"))).ink_fraction == 0.0

    def test_ink_is_measured_against_the_background_not_transparency(self):
        """A painted backdrop must not read as a full canvas of ink."""
        source = svg(
            '<rect width="100" height="100" fill="#eef"/><circle cx="50" cy="50" r="10"/>'
        )
        assert image_stats(render_png(source)).ink_fraction < 0.2

    def test_black_on_white_counts_as_ink(self):
        """Squared channel differences overflow int16 and silently vanish."""
        source = svg('<rect x="0" y="0" width="100" height="50" fill="black"/>')
        assert image_stats(render_png(source)).ink_fraction > 0.4


class TestGeometry:
    """Shape extraction must not depend on how the shape was spelled."""

    CIRCLES = {
        "circle": '<circle cx="50" cy="50" r="25"/>',
        "bezier": (
            '<path d="M 75 50 C 75 63.8 63.8 75 50 75 C 36.2 75 25 63.8 25 50 '
            'C 25 36.2 36.2 25 50 25 C 63.8 25 75 36.2 75 50 Z"/>'
        ),
        "arc": '<path d="M 25 50 A 25 25 0 1 1 75 50 A 25 25 0 1 1 25 50 Z"/>',
        # The same two arcs with the flags run together and into the next
        # coordinate, which the spec allows and minifiers produce.
        "arc_compact_flags": '<path d="M 25 50 A 25 25 0 1175 50 A 25 25 0 1125 50 Z"/>',
    }

    @pytest.mark.parametrize("spelling", sorted(CIRCLES))
    def test_every_circle_spelling_agrees(self, spelling):
        shape = extract_shapes(parse_svg(svg(self.CIRCLES[spelling])))[0]
        assert shape.circularity == pytest.approx(1.0, abs=0.01)
        assert shape.radius == pytest.approx(0.25, abs=0.005)
        assert shape.centroid == pytest.approx((0.5, 0.5), abs=0.005)

    def test_square_is_not_round(self):
        shape = extract_shapes(
            parse_svg(svg('<rect x="25" y="25" width="50" height="50"/>'))
        )[0]
        assert shape.circularity == pytest.approx(math.pi / 4, abs=0.01)

    def test_centroid_is_not_biased_by_vertex_count(self):
        """A duplicated closing vertex used to drag the centroid off centre."""
        shape = extract_shapes(
            parse_svg(svg('<rect x="25" y="25" width="50" height="50"/>'))
        )[0]
        assert shape.centroid == pytest.approx((0.5, 0.5), abs=0.001)

    def test_radius_variation_survives_coarse_sampling(self):
        """With four raw segments a square's radius variation reads as zero."""
        shape = extract_shapes(
            parse_svg(svg('<rect x="25" y="25" width="50" height="50"/>'))
        )[0]
        assert shape.radius_cv == pytest.approx(0.107, abs=0.02)

    def test_smooth_cubic_after_quadratic_does_not_reflect_its_control(self):
        """S may only reflect after C/S; after Q its first control is the cursor."""
        points = [
            p
            for sub, _ in flatten_path("M 0 0 Q 50 100 100 0 S 200 100 200 0")
            for p in sub
        ]
        assert min(y for _, y in points) >= 0

    def test_smooth_quadratic_after_cubic_does_not_reflect_its_control(self):
        """T may only reflect after Q/T; after C its control is the cursor."""
        points = [
            p
            for sub, _ in flatten_path("M 0 0 C 20 80 80 80 100 0 T 200 0")
            for p in sub
        ]
        assert min(y for _, y in points) >= 0

    @pytest.mark.parametrize(
        "hiding",
        [
            'display="none"',
            'style="display: none"',
            'visibility="hidden"',
            'opacity="0"',
        ],
    )
    def test_hidden_geometry_is_not_scored(self, hiding):
        source = svg(f'<circle cx="50" cy="50" r="25" {hiding}/>')
        assert extract_shapes(parse_svg(source)) == []

    def test_a_hidden_group_hides_its_children(self):
        source = svg('<g display="none"><circle cx="50" cy="50" r="25"/></g>')
        assert extract_shapes(parse_svg(source)) == []

    def test_unpainted_geometry_is_not_scored(self):
        source = svg('<circle cx="50" cy="50" r="25" fill="none" stroke="none"/>')
        assert extract_shapes(parse_svg(source)) == []

    def test_a_stroked_outline_with_no_fill_still_counts(self):
        source = svg('<circle cx="50" cy="50" r="25" fill="none" stroke="black"/>')
        assert len(extract_shapes(parse_svg(source))) == 1

    def test_a_child_can_repaint_inside_an_unpainted_group(self):
        source = svg(
            '<g fill="none" stroke="none"><circle cx="50" cy="50" r="25" fill="black"/></g>'
        )
        assert len(extract_shapes(parse_svg(source))) == 1

    def test_off_canvas_geometry_is_not_scored(self):
        source = svg('<circle cx="-500" cy="50" r="25"/>')
        assert extract_shapes(parse_svg(source)) == []

    def test_partially_visible_geometry_still_counts(self):
        source = svg('<circle cx="0" cy="50" r="25"/>')
        assert len(extract_shapes(parse_svg(source))) == 1

    def test_long_ellipse_is_rejected_by_aspect(self):
        shape = extract_shapes(
            parse_svg(svg('<ellipse cx="50" cy="50" rx="30" ry="15"/>'))
        )[0]
        assert shape.aspect == pytest.approx(2.0, abs=0.05)

    def test_group_transforms_are_applied(self):
        source = svg(
            '<g transform="translate(20 20)"><circle cx="30" cy="30" r="25"/></g>'
        )
        shape = extract_shapes(parse_svg(source))[0]
        assert shape.centroid == pytest.approx((0.5, 0.5), abs=0.005)

    def test_transform_composition_is_left_to_right(self):
        assert apply(parse_transform("translate(10 0) scale(2)"), (5.0, 5.0)) == (
            20.0,
            10.0,
        )
        assert apply(parse_transform("scale(2) translate(10 0)"), (5.0, 5.0)) == (
            30.0,
            10.0,
        )

    def test_rotate_about_a_point(self):
        x, y = apply(parse_transform("rotate(90)"), (1.0, 0.0))
        assert (x, y) == pytest.approx((0.0, 1.0), abs=1e-9)


class TestLengthUnits:
    """Percentages and CSS units are legal in every geometry attribute.

    Models use them freely, `width="100%"` most of all, so the raw attribute
    cannot be passed to float().
    """

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("12", 12.0),
            ("12px", 12.0),
            ("100%", 400.0),
            ("50%", 200.0),
            ("72pt", 96.0),
            ("1in", 96.0),
            ("1pc", 16.0),
            ("25.4mm", 96.0),
            ("2.54cm", 96.0),
            ("1em", 16.0),
            ("5e1", 50.0),
            (" 12 ", 12.0),
        ],
    )
    def test_units_resolve(self, value, expected):
        assert length(value, reference=400.0) == pytest.approx(expected)

    @pytest.mark.parametrize("value", ["garbage", "", None, "12 34", "calc(1px)"])
    def test_unparseable_falls_back_to_the_default(self, value):
        assert length(value, reference=400.0, default=7.0) == 7.0

    @pytest.mark.parametrize(
        "body",
        [
            '<rect width="100%" height="100%" fill="#eef"/>',
            '<circle cx="50%" cy="50%" r="10%"/>',
            '<rect x="10px" y="10px" width="50px" height="50px"/>',
            '<circle cx="20pt" cy="20pt" r="5pt"/>',
            '<rect x="1em" y="1em" width="2em" height="2em"/>',
            '<line x1="0%" y1="0%" x2="100%" y2="100%"/>',
            '<circle cx="wat" cy="nope" r="10"/>',
        ],
    )
    def test_extraction_survives_units(self, body):
        assert extract_shapes(parse_svg(svg(body, "0 0 400 300"))) is not None

    def test_percentages_resolve_against_the_matching_axis(self):
        """Width percentages use viewport width, height percentages the height."""
        shape = extract_shapes(
            parse_svg(
                svg('<rect x="0" y="0" width="50%" height="50%"/>', "0 0 400 200")
            )
        )[0]
        # Normalisation divides by the longer side, 400.
        assert shape.bbox == pytest.approx((0.0, 0.0, 0.5, 0.25), abs=0.001)


class TestStructure:
    """Structural checks over the extracted shapes."""

    def structure(self, name, expected_wheels=2):
        return analyse_structure(
            extract_shapes(parse_svg(fixture(name))), expected_wheels=expected_wheels
        )

    def test_complete_scene_passes_everything(self):
        assert self.structure("good_pelican_bike").score == 1.0

    def test_wheels_drawn_as_paths_score_the_same(self):
        """Representation must not change the structural verdict."""
        assert self.structure("wheels_as_paths").score == 1.0

    def test_wheels_placed_with_use_score_the_same(self):
        """A model that factors its SVG must not be punished for it.

        Nothing inside `<defs>` is painted and every `<use>` is, so the
        wheels are at the positions the `<use>` elements give them.
        """
        report = self.structure("wheels_via_use")
        assert report.score == 1.0
        assert [round(w.centroid[0], 3) for w in report.wheels] == [0.325, 0.675]

    def test_definitions_alone_draw_nothing(self):
        """`<defs>` content is a template, not paint. Without a `<use>` to
        instantiate it there is no geometry on the canvas at all."""
        source = svg(
            '<defs><circle id="w" cx="20" cy="80" r="15"/>'
            '<circle id="x" cx="80" cy="80" r="15"/></defs>',
        )
        assert extract_shapes(parse_svg(source)) == []

    def test_background_rect_is_not_mistaken_for_a_rider(self):
        report = self.structure("good_pelican_bike")
        assert report.rider is not None
        assert report.rider.area < 0.35

    def test_animal_head_is_not_counted_as_a_wheel(self):
        """Any circle passes a roundness test, so wheels are found by row."""
        assert len(self.structure("good_pelican_bike").wheels) == 2

    def test_vehicle_without_rider_loses_only_the_rider_checks(self):
        report = self.structure("bike_no_bird")
        assert not report.checks[-1].passed
        assert 0.0 < report.score < 1.0
        assert len(report.wheels) == 2

    def test_rider_without_vehicle_scores_zero(self):
        assert self.structure("bird_no_bike").score == 0.0

    def test_scribble_scores_zero(self):
        assert self.structure("bad_scribble").score == 0.0

    def test_a_pedal_below_the_axle_does_not_displace_a_single_wheel(self):
        """With one expected wheel, rows are ranked by bottom edge.

        A pedal drawn below the axle has a lower centre than the wheel, but
        the wheel's rim is what reaches the ground.
        """
        source = svg(
            '<circle cx="200" cy="200" r="60" fill="none" stroke="#222" stroke-width="6"/>'
            '<circle cx="230" cy="245" r="14" fill="#222"/>'
            '<line x1="200" y1="140" x2="200" y2="80" stroke="#222" stroke-width="6"/>',
            view_box="0 0 400 300",
        )
        wheels = find_wheels(
            significant_shapes(extract_shapes(parse_svg(source))), expected=1
        )
        assert len(wheels) == 1
        assert wheels[0].radius == pytest.approx(0.15, abs=0.01)

    def test_frame_check_spans_the_outermost_wheels(self):
        """A third wheel-like shape in the row must not shrink the span.

        The frame connects the outermost hubs, and anything sitting between
        them, another wheel candidate included, counts as bridging geometry.
        """
        source = svg(
            '<circle cx="90" cy="220" r="34" fill="none" stroke="#222" stroke-width="5"/>'
            '<circle cx="180" cy="220" r="45" fill="none" stroke="#222" stroke-width="5"/>'
            '<circle cx="380" cy="220" r="45" fill="none" stroke="#222" stroke-width="5"/>'
            '<line x1="180" y1="220" x2="280" y2="170" stroke="#222" stroke-width="5"/>'
            '<line x1="280" y1="170" x2="380" y2="220" stroke="#222" stroke-width="5"/>'
            '<line x1="180" y1="220" x2="380" y2="220" stroke="#222" stroke-width="5"/>'
            '<ellipse cx="280" cy="130" rx="45" ry="30" fill="gray"/>',
            view_box="0 0 500 300",
        )
        shapes = significant_shapes(extract_shapes(parse_svg(source)))
        assert len(find_wheels(shapes, expected=2)) == 3
        report = analyse_structure(shapes, expected_wheels=2).to_dict()
        assert report["checks"]["frame_spans_wheels"] is True

    def test_two_real_wheels_beat_a_pedal_when_one_wheel_is_expected(self):
        """A bicycle scored as a unicycle must fail wheel_count.

        The expected count is a preference and the ground is a fact, so the
        two wheels win the row ranking and the check then fails honestly,
        rather than the pedal being adopted as the single expected wheel.
        """
        source = svg(
            '<circle cx="120" cy="220" r="55" fill="none" stroke="#222" stroke-width="6"/>'
            '<circle cx="280" cy="220" r="55" fill="none" stroke="#222" stroke-width="6"/>'
            '<circle cx="200" cy="250" r="14" fill="#222"/>'
            '<line x1="120" y1="220" x2="280" y2="220" stroke="#222" stroke-width="6"/>',
            view_box="0 0 400 300",
        )
        shapes = significant_shapes(extract_shapes(parse_svg(source)))
        wheels = find_wheels(shapes, expected=1)
        assert len(wheels) == 2
        report = analyse_structure(shapes, expected_wheels=1)
        assert report.to_dict()["checks"]["wheel_count"] is False

    def test_a_rider_bigger_than_the_wheels_does_not_hide_them(self):
        """A body drawn larger than the wheels must not become the anchor.

        Wheels are found as a row, so the largest round shape in the drawing
        cannot displace them however big it is.
        """
        body_radius, wheel_radius = 58, 52
        source = svg(
            f'<circle cx="110" cy="232" r="{wheel_radius}" fill="none" stroke="#222" stroke-width="6"/>'
            f'<circle cx="290" cy="232" r="{wheel_radius}" fill="none" stroke="#222" stroke-width="6"/>'
            '<line x1="110" y1="232" x2="290" y2="232" stroke="#c00" stroke-width="6"/>'
            '<line x1="110" y1="232" x2="200" y2="160" stroke="#c00" stroke-width="6"/>'
            '<line x1="290" y1="232" x2="200" y2="160" stroke="#c00" stroke-width="6"/>'
            f'<circle cx="200" cy="110" r="{body_radius}" fill="#fff" stroke="#333" stroke-width="3"/>',
            "0 0 400 300",
        )
        report = analyse_structure(extract_shapes(parse_svg(source)), expected_wheels=2)
        assert len(report.wheels) == 2, "the body displaced the wheels"
        assert report.score == 1.0

    def test_wheels_are_taken_from_the_lower_row(self):
        """Two round shapes level at the top must not be read as the wheels."""
        source = svg(
            '<circle cx="120" cy="90" r="40" fill="#fff" stroke="#333" stroke-width="3"/>'
            '<circle cx="280" cy="90" r="40" fill="#fff" stroke="#333" stroke-width="3"/>'
            '<circle cx="110" cy="232" r="52" fill="none" stroke="#222" stroke-width="6"/>'
            '<circle cx="290" cy="232" r="52" fill="none" stroke="#222" stroke-width="6"/>'
            '<line x1="110" y1="232" x2="290" y2="232" stroke="#c00" stroke-width="6"/>',
            "0 0 400 300",
        )
        wheels = analyse_structure(
            extract_shapes(parse_svg(source)), expected_wheels=2
        ).wheels
        assert len(wheels) == 2
        assert all(w.centroid[1] > 0.5 for w in wheels), "picked the upper row"

    def test_a_hub_does_not_outrank_the_rim_it_sits_in(self):
        """The bug that made the same drawing score differently on two machines.

        A hub is concentric with its rim but too small to share a row with it, so
        it forms a valid row of one at the same height. The row comparison used
        raw floats, so the two rows differed in the last bits of `cy` and the tie
        was settled before the radius was consulted: a hub of radius 0.02 beat a
        rim of 0.1248, the axle line jumped up the canvas and the rider fell
        outside it. Locally the drawing scored 1.000 and on the deployed Space
        0.333.
        """
        source = svg(
            '<circle cx="200" cy="280" r="50" fill="none" stroke="#222" stroke-width="6"/>'
            '<circle cx="200" cy="280" r="46" fill="none" stroke="#777" stroke-width="2"/>'
            '<circle cx="200" cy="280" r="8" fill="#444" stroke="#222" stroke-width="2"/>'
            '<line x1="200" y1="230" x2="200" y2="150" stroke="#888" stroke-width="5"/>'
            '<ellipse cx="170" cy="110" rx="55" ry="40" fill="#8a5a2b" stroke="#333" stroke-width="3"/>',
            "0 0 400 400",
        )
        report = analyse_structure(extract_shapes(parse_svg(source)), expected_wheels=1)
        assert len(report.wheels) == 1
        assert report.wheels[0].radius > 0.1, "the hub was taken for the wheel"
        assert report.rider is not None
        assert report.score == 1.0

    def test_row_choice_survives_last_bit_noise_in_height(self):
        """Whatever wins must not be decided by floating-point rounding."""
        import math as _math
        from dataclasses import replace as _replace

        source = svg(
            '<circle cx="200" cy="280" r="50" fill="none" stroke="#222"/>'
            '<circle cx="200" cy="280" r="8" fill="#444"/>'
            '<ellipse cx="170" cy="110" rx="55" ry="40" fill="#8a5a2b"/>',
            "0 0 400 400",
        )
        shapes = significant_shapes(extract_shapes(parse_svg(source)))
        radii = set()
        for nudge in (0, 1, -1, 2, -2):
            moved = []
            for shape in shapes:
                cy = shape.centroid[1]
                for _ in range(abs(nudge)):
                    cy = _math.nextafter(cy, _math.inf if nudge > 0 else -_math.inf)
                moved.append(_replace(shape, centroid=(shape.centroid[0], cy)))
            wheels = find_wheels(moved, expected=1)
            radii.add(round(wheels[0].radius, 4) if wheels else None)
        assert len(radii) == 1, f"the choice moved with float noise: {radii}"

    def test_single_wheel_vehicle_drops_the_pair_checks(self):
        """Scoring a unicycle against bicycle geometry punished a good drawing."""
        names = {
            c.name for c in self.structure("bike_no_bird", expected_wheels=1).checks
        }
        assert "wheels_apart" not in names
        assert "wheel_count" in names


class TestGate:
    """The full deterministic admission check."""

    @pytest.mark.parametrize("name", sorted(p.stem for p in FIXTURES.glob("*.svg")))
    def test_corpus_admission_matches_ground_truth(self, name):
        result = run_gate(fixture(name), PELICAN_TERMS)
        assert result.passed is (name in ADMISSIBLE), result.codes

    def test_off_canvas_is_distinguished_from_blank(self):
        assert "content_off_canvas" in run_gate(fixture("offcanvas")).codes
        assert "blank_canvas" in run_gate(fixture("blank")).codes

    def test_passing_gate_returns_the_raster_for_reuse(self):
        """The submission is rasterised once and the judge reuses the bytes."""
        assert run_gate(fixture("good_pelican_bike"), PELICAN_TERMS).png is not None


class TestTasks:
    """The subject-by-vehicle grid."""

    def test_only_the_canonical_pair_is_not_held_out(self):
        not_held_out = [t.task_id for t in all_tasks() if not t.held_out]
        assert not_held_out == ["pelican_bicycle"]

    def test_held_out_only_excludes_the_canonical_pair(self):
        assert all(t.held_out for t in all_tasks(held_out_only=True))

    def test_sampling_is_reproducible_under_a_seed(self):
        assert sample_task(seed=7).task_id == sample_task(seed=7).task_id

    def test_prompt_forbids_the_two_known_cheats(self):
        prompt = make_task("pelican", "bicycle").prompt
        assert "raster" in prompt and "text" in prompt

    def test_forbidden_terms_cover_subject_and_vehicle(self):
        terms = make_task("pelican", "bicycle").forbidden_terms
        assert "pelican" in terms and "bicycle" in terms and "bird" in terms

    def test_unknown_task_id_names_the_valid_ones(self):
        with pytest.raises(ValueError, match="pelican_bicycle"):
            task_from_ids(["pelican_bicycel"])


class StubVisionClient:
    """A judge endpoint that answers from a script instead of a network."""

    def __init__(self, caption="a pelican riding a bicycle", answer=True, fail=False):
        self.model = "stub"
        self.caption = caption
        self.answer = answer
        self.fail = fail
        self.calls = 0

    async def complete_with_image(
        self, prompt, png_bytes, *, schema=None, max_tokens=400
    ):
        self.calls += 1
        if self.fail:
            raise RuntimeError("judge is down")
        if schema is None:
            return self.caption
        keys = list(schema["properties"])
        return "{" + ", ".join(f'"{k}": {str(self.answer).lower()}' for k in keys) + "}"


class TestScoring:
    """Composition of the layers into one reward."""

    task = make_task("pelican", "bicycle")

    def evaluate(self, name, judge=None):
        return asyncio.run(
            evaluate_submission(f"```svg\n{fixture(name)}\n```", self.task, judge)
        )

    def test_rejected_submission_scores_zero(self):
        assert self.evaluate("cheat_raster_datauri").reward == 0.0

    def test_offline_mode_scores_on_structure_alone(self):
        evaluation = self.evaluate("good_pelican_bike")
        assert not evaluation.judge_enabled
        assert evaluation.reward == pytest.approx(evaluation.structure_score)

    def test_judge_failure_does_not_raise_the_reward(self):
        """Otherwise breaking the judge becomes a winning strategy."""
        broken = self.evaluate(
            "good_pelican_bike", VisionJudge(StubVisionClient(fail=True))
        )
        assert not broken.judged
        assert broken.reward < self.evaluate("good_pelican_bike").reward

    def test_perfect_judge_gives_full_marks(self):
        evaluation = self.evaluate(
            "good_pelican_bike", VisionJudge(StubVisionClient(answer=True))
        )
        assert evaluation.reward == pytest.approx(1.0)

    def test_blind_caption_that_misses_the_subject_costs_marks(self):
        named = self.evaluate(
            "good_pelican_bike",
            VisionJudge(StubVisionClient(caption="a pelican on a bicycle")),
        )
        unnamed = self.evaluate(
            "good_pelican_bike", VisionJudge(StubVisionClient(caption="some shapes"))
        )
        assert unnamed.reward < named.reward

    def test_not_riding_is_penalised_but_not_erased(self):
        judge = VisionJudge(StubVisionClient(answer=True))
        riding = self.evaluate("good_pelican_bike", judge)
        # Same drawing, judge says nothing is riding.
        not_riding = asyncio.run(
            evaluate_submission(
                f"```svg\n{fixture('good_pelican_bike')}\n```",
                self.task,
                VisionJudge(_NotRidingClient()),
            )
        )
        assert 0.0 < not_riding.semantic_score < riding.semantic_score

    def test_truncated_reply_is_reported_as_such(self):
        evaluation = evaluate_deterministic(
            "<svg xmlns='http://www.w3.org/2000/svg'><circle r='1'/>", self.task
        )
        assert evaluation.gate.codes == ["truncated_svg"]


class _NotRidingClient(StubVisionClient):
    """Answers yes to everything except the riding posture."""

    async def complete_with_image(
        self, prompt, png_bytes, *, schema=None, max_tokens=400
    ):
        if schema is None:
            return self.caption
        keys = list(schema["properties"])
        parts = [f'"{k}": {"false" if k == "riding_posture" else "true"}' for k in keys]
        return "{" + ", ".join(parts) + "}"


class TestRewardConsistency:
    """The rubric tree and the evaluation must never disagree."""

    @pytest.mark.parametrize("judge_enabled", [True, False])
    def test_weights_come_from_one_place(self, judge_enabled):
        structure_weight, semantic_weight = component_weights(judge_enabled)
        assert structure_weight + semantic_weight == pytest.approx(1.0)

    @pytest.mark.parametrize("name", sorted(ADMISSIBLE))
    def test_rubric_matches_evaluation_reward(self, name):
        environment = PelicanSvgEnvironment(
            subject="pelican", vehicle="bicycle", enable_judge=False
        )
        environment.reset()
        observation = environment.step(PelicanSvgAction(response=fixture(name)))
        expected = build_rubric(judge_enabled=False)(None, observation)
        assert observation.reward == pytest.approx(expected)


class TestEnvironment:
    """The Gym-shaped contract."""

    def environment(self, **kwargs):
        kwargs.setdefault("enable_judge", False)
        return PelicanSvgEnvironment(**kwargs)

    def test_reset_returns_a_prompt_and_no_reward(self):
        observation = self.environment().reset(seed=3)
        assert observation.prompt
        assert observation.reward is None
        assert not observation.done

    def test_step_before_reset_raises(self):
        with pytest.raises(RuntimeError, match="reset"):
            self.environment().step(PelicanSvgAction(response=fixture("blank")))

    def test_step_ends_the_episode(self):
        environment = self.environment(subject="pelican", vehicle="bicycle")
        environment.reset()
        observation = environment.step(
            PelicanSvgAction(response=fixture("good_pelican_bike"))
        )
        assert observation.done
        assert observation.reward == pytest.approx(1.0)
        assert environment.state.submitted

    def test_step_async_scores_off_the_event_loop(self, monkeypatch):
        """Scoring is CPU-bound, so it must not stall other sessions sharing
        the server's event loop."""
        from envs.pelican_svg_env.server import scoring

        loop_ran = threading.Event()
        evaluate_deterministic = scoring.evaluate_deterministic

        def evaluate_once_the_loop_ran(response, task):
            assert loop_ran.wait(timeout=5), "scoring blocked the event loop"
            return evaluate_deterministic(response, task)

        monkeypatch.setattr(
            scoring, "evaluate_deterministic", evaluate_once_the_loop_ran
        )
        environment = self.environment(subject="pelican", vehicle="bicycle")
        environment.reset()

        async def other_session():
            loop_ran.set()

        async def run():
            action = PelicanSvgAction(response=fixture("good_pelican_bike"))
            observation, _ = await asyncio.gather(
                environment.step_async(action), other_session()
            )
            return observation

        assert asyncio.run(run()).reward == pytest.approx(1.0)

    def test_pelican_svg_web_playground(self):
        environment = self.environment(subject="pelican", vehicle="bicycle")
        task = environment.render_web(environment.reset().model_dump())
        assert "pelican" in task
        assert "#" not in task  # theme colours only, so it reads in dark mode
        drawing = environment.render_web(
            environment.step(
                PelicanSvgAction(response=fixture("good_pelican_bike"))
            ).model_dump()
        )
        assert (
            '<img alt="Pelican SVG drawing" src="data:image/svg+xml;base64,' in drawing
        )
        assert "<svg" not in drawing  # never inlined, so its scripts can't run
        assert "reward 1.00" in drawing
        assert environment.render_web({}) is None

    def test_task_can_be_pinned_by_id(self):
        """A benchmark run must ask every model the same question."""
        observation = self.environment().reset(task_id="capybara_unicycle")
        assert observation.task_id == "capybara_unicycle"
        assert observation.expected_wheels == 1

    def test_truthy_strings_do_not_pass_the_checklist(self):
        """Only a JSON true counts. A judge that answers "false" as a string
        ignored the schema, and that must not read as approval."""
        from envs.pelican_svg_env.server.vision_judge import _parse_checklist

        questions = {"a": "?", "b": "?", "c": "?", "d": "?"}
        raw = '{"a": "false", "b": "no", "c": "0", "d": true}'
        assert _parse_checklist(raw, questions) == {
            "a": False,
            "b": False,
            "c": False,
            "d": True,
        }

    def test_partial_pin_is_honoured(self):
        """Pinning one dimension must not be silently ignored."""
        assert self.environment(subject="octopus").reset().task_id == "octopus_bicycle"
        observation = self.environment(vehicle="skateboard").reset()
        assert observation.task_id == "pelican_skateboard"
        assert observation.expected_wheels == 2

    def test_partial_pin_works_at_reset_time_too(self):
        environment = self.environment()
        assert environment.reset(subject="octopus").task_id == "octopus_bicycle"
        assert environment.reset(vehicle="skateboard").task_id == "pelican_skateboard"

    def test_pin_takes_precedence_over_sampling(self):
        environment = self.environment(sample_tasks=True, subject="octopus")
        assert environment.reset(seed=5).task_id == "octopus_bicycle"

    def test_held_out_only_rejects_a_pin_that_resolves_canonical(self):
        """held_out_only is a hard promise, not a default."""
        with pytest.raises(ValueError, match="held-out"):
            self.environment(held_out_only=True, subject="pelican").reset()
        with pytest.raises(ValueError, match="held-out"):
            self.environment(held_out_only=True).reset(task_id="pelican_bicycle")

    def test_held_out_only_accepts_a_held_out_pin(self):
        environment = self.environment(held_out_only=True, subject="capybara")
        assert environment.reset().task_id == "capybara_bicycle"

    def test_default_task_is_the_original_prompt(self):
        """A default that changes per reset makes two runs incomparable."""
        assert self.environment().reset().task_id == "pelican_bicycle"
        assert self.environment().reset(seed=99).task_id == "pelican_bicycle"

    def test_seeded_sampling_is_reproducible(self):
        first = self.environment(sample_tasks=True).reset(seed=11).task_id
        second = self.environment(sample_tasks=True).reset(seed=11).task_id
        assert first == second

    def test_held_out_only_never_serves_the_original_prompt(self):
        ids = {
            self.environment(held_out_only=True).reset(seed=s).task_id
            for s in range(25)
        }
        assert "pelican_bicycle" not in ids
        assert len(ids) > 1

    def test_violations_are_surfaced_to_the_caller(self):
        environment = self.environment(subject="pelican", vehicle="bicycle")
        environment.reset()
        observation = environment.step(
            PelicanSvgAction(response=fixture("cheat_raster_datauri"))
        )
        assert "embedded_raster" in observation.violations
        assert observation.reward == 0.0


class TestJudgeConfiguration:
    """Which judge `_judge_from_env` builds, and what it is allowed to send."""

    @staticmethod
    def _clear(monkeypatch):
        for name in (
            "PELICAN_SVG_DISABLE_JUDGE",
            "PELICAN_SVG_JUDGE_MODEL",
            "PELICAN_SVG_JUDGE_BASE_URL",
            "PELICAN_SVG_JUDGE_API_KEY",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(
            "envs.pelican_svg_env.server.pelican_svg_environment.get_token",
            lambda: None,
        )

    def test_no_token_and_no_endpoint_runs_offline(self, monkeypatch):
        self._clear(monkeypatch)
        assert _judge_from_env() is None

    def test_endpoint_enables_the_judge_without_a_token(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("PELICAN_SVG_JUDGE_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("PELICAN_SVG_JUDGE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

        judge = _judge_from_env()

        assert judge is not None
        assert judge._client.base_url == "http://localhost:8000/v1"
        assert judge._client.model == "Qwen/Qwen2.5-VL-7B-Instruct"

    def test_endpoint_without_a_model_is_an_error(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("PELICAN_SVG_JUDGE_BASE_URL", "http://localhost:8000/v1")

        with pytest.raises(ValueError, match="PELICAN_SVG_JUDGE_MODEL"):
            _judge_from_env()

    def test_disable_wins_over_an_endpoint(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("PELICAN_SVG_JUDGE_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("PELICAN_SVG_JUDGE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
        monkeypatch.setenv("PELICAN_SVG_DISABLE_JUDGE", "1")

        assert _judge_from_env() is None

    def test_hosted_judge_is_unchanged(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("HF_TOKEN", "hf_token_value")

        judge = _judge_from_env()

        assert judge is not None
        assert judge._client.base_url is None
        assert judge._client.model == DEFAULT_JUDGE_MODEL

    def test_hugging_face_token_is_never_sent_to_an_endpoint(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("HF_TOKEN", "hf_token_value")
        monkeypatch.setenv("PELICAN_SVG_JUDGE_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("PELICAN_SVG_JUDGE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

        client = _judge_from_env()._client

        assert "hf_token_value" not in str(vars(client).values())
        assert client._api_key == "EMPTY"

    def test_server_refuses_to_start_with_an_endpoint_but_no_model(self, monkeypatch):
        """Otherwise every session fails with an opaque factory error."""
        self._clear(monkeypatch)
        monkeypatch.setenv("PELICAN_SVG_JUDGE_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.delitem(
            sys.modules, "envs.pelican_svg_env.server.app", raising=False
        )

        with pytest.raises(ValueError, match="PELICAN_SVG_JUDGE_MODEL"):
            importlib.import_module("envs.pelican_svg_env.server.app")

    def test_server_starts_normally_with_a_hosted_judge(self, monkeypatch):
        """The startup check must not get in the way of the default setup."""
        self._clear(monkeypatch)
        monkeypatch.setenv("HF_TOKEN", "hf_token_value")
        monkeypatch.delitem(
            sys.modules, "envs.pelican_svg_env.server.app", raising=False
        )

        module = importlib.import_module("envs.pelican_svg_env.server.app")

        assert module.app is not None

    def test_endpoint_api_key_is_used_when_given(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("PELICAN_SVG_JUDGE_BASE_URL", "http://gateway/v1")
        monkeypatch.setenv("PELICAN_SVG_JUDGE_MODEL", "vision-model")
        monkeypatch.setenv("PELICAN_SVG_JUDGE_API_KEY", "sk-endpoint")

        assert _judge_from_env()._client._api_key == "sk-endpoint"


class TestVisionClientRequest:
    """What `HFVisionClient` hands to the inference client."""

    @staticmethod
    def _capture(monkeypatch):
        seen: dict[str, object] = {}

        class FakeClient:
            def __init__(self, **kwargs):
                seen["init"] = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def chat_completion(self, **kwargs):
                seen["call"] = kwargs
                message = type("M", (), {"content": "a pelican"})
                choice = type("C", (), {"message": message})
                return type("R", (), {"choices": [choice]})

        monkeypatch.setattr(
            "huggingface_hub.AsyncInferenceClient", FakeClient, raising=False
        )
        return seen

    def test_hosted_client_gets_no_base_url(self, monkeypatch):
        seen = self._capture(monkeypatch)
        client = HFVisionClient(model="model-id", api_key="hf_token_value")

        asyncio.run(client.complete_with_image("what is this?", b"png"))

        assert "base_url" not in seen["init"]
        assert seen["init"]["api_key"] == "hf_token_value"
        assert seen["call"]["model"] == "model-id"

    def test_endpoint_client_routes_to_the_base_url(self, monkeypatch):
        seen = self._capture(monkeypatch)
        client = HFVisionClient(
            model="served-model", base_url="http://localhost:8000/v1"
        )

        asyncio.run(client.complete_with_image("what is this?", b"png"))

        assert seen["init"]["base_url"] == "http://localhost:8000/v1"
        # The endpoint reads the model from the body, not from the URL.
        assert seen["call"]["model"] == "served-model"
        assert seen["call"]["temperature"] == 0.0
