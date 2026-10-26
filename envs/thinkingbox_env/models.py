"""Define the public action, observation, and state models for ThinkingBox.

These Pydantic models are the only benchmark-specific values serialized across
the OpenEnv WebSocket boundary.
"""

from typing import Annotated, Any, Literal

from openenv.core import Action, ListToolsAction, Observation, State, Tool
from pydantic import BaseModel, ConfigDict, Field, model_validator, TypeAdapter


DATA_RELEASE_NAME = "thinkingbox-bench-v1.0"
DATA_COMMIT = "fcaba4c1a9debec42fda7f15bf29fe6d6b46c431"
DATA_BUNDLE_SHA256 = "16803d95d161ca06309d95f0dbbcbbb7b3fdf8a61e2b4eacfe920781b12f18ff"
DATA_MANIFEST_PATH = "releases/thinkingbox_bench_v1/testlist_thinkingbox_bench_v1.yaml"
DATA_ARCHIVE_URL = (
    f"https://codeload.github.com/microsoft/thinkingbox-data/tar.gz/{DATA_COMMIT}"
)


# Generated from the commit-pinned public benchmark manifest.
CANONICAL_TEST_UIDS: tuple[str, ...] = (
    "sandbox_external_retail_group1.py:test_case_ST002_001",
    "sandbox_external_retail_group1.py:test_case_ST002_003",
    "sandbox_external_retail_group1.py:test_case_ST002_004",
    "sandbox_external_retail_group1.py:test_case_ST003_003",
    "sandbox_external_retail_group1.py:test_case_ST003_004",
    "sandbox_external_retail_group1.py:test_case_ST003_006",
    "sandbox_external_retail_group1.py:test_case_ST004_001",
    "sandbox_external_retail_group1.py:test_case_ST004_003",
    "sandbox_external_retail_group1.py:test_case_ST004_004",
    "sandbox_external_retail_group1.py:test_case_ST004_005",
    "sandbox_external_retail_group1.py:test_case_ST004_007",
    "sandbox_external_retail_group1.py:test_case_ST004_008",
    "sandbox_external_retail_group1.py:test_case_ST004_009",
    "sandbox_external_retail_group1.py:test_case_ST005_001",
    "sandbox_external_retail_group1.py:test_case_ST005_005",
    "sandbox_external_retail_group1.py:test_case_ST006_001",
    "sandbox_external_retail_group1.py:test_case_ST006_002",
    "sandbox_external_retail_group1.py:test_case_ST006_003",
    "sandbox_external_retail_group1.py:test_case_ST006_004",
    "sandbox_external_retail_group1.py:test_case_ST006_005",
    "sandbox_external_retail_group1.py:test_case_ST006_006",
    "sandbox_external_retail_group1.py:test_case_ST006_008",
    "sandbox_external_retail_group1.py:test_case_ST006_010",
    "sandbox_external_retail_group1.py:test_case_ST006_011",
    "sandbox_external_retail_group1.py:test_case_ST006_015",
    "sandbox_external_retail_group1.py:test_case_ST006_016",
    "sandbox_external_retail_group1.py:test_case_ST006_017",
    "sandbox_external_retail_group1.py:test_case_ST006_019",
    "sandbox_external_retail_group1.py:test_case_ST006_021",
    "sandbox_external_retail_group1.py:test_case_ST006_022",
    "sandbox_external_retail_group1.py:test_case_ST006_023",
    "sandbox_external_retail_group1.py:test_case_ST006_024",
    "sandbox_external_retail_group1.py:test_case_ST006_025",
    "sandbox_external_retail_group1.py:test_case_ST006_026",
    "sandbox_external_retail_group1.py:test_case_ST006_028",
    "sandbox_external_retail_group1.py:test_case_ST006_030",
    "sandbox_external_retail_group1.py:test_case_ST006_031",
    "sandbox_external_retail_group1.py:test_case_ST006_032",
    "sandbox_external_retail_group1.py:test_case_ST006_033",
    "sandbox_external_retail_group1.py:test_case_ST006_034",
    "sandbox_external_retail_group1.py:test_case_ST006_037",
    "sandbox_external_retail_group1.py:test_case_ST006_039",
    "sandbox_external_retail_group1.py:test_case_ST006_040",
    "sandbox_external_retail_group1.py:test_case_ST006_041",
    "sandbox_external_retail_group1.py:test_case_ST006_042",
    "sandbox_external_retail_group1.py:test_case_ST006_043",
    "sandbox_external_retail_group1.py:test_case_ST006_044",
    "sandbox_external_retail_group1.py:test_case_ST006_045",
    "sandbox_external_retail_group1.py:test_case_ST006_047",
    "sandbox_external_retail_group1.py:test_case_ST006_049",
    "sandbox_external_retail_group1.py:test_case_ST007_003",
    "sandbox_external_retail_group1.py:test_case_ST007_005",
    "sandbox_external_retail_group1.py:test_case_ST007_006",
    "sandbox_external_retail_group1.py:test_case_ST007_008",
    "sandbox_external_retail_group1.py:test_case_ST008_003",
    "sandbox_external_retail_group1.py:test_case_ST009_001",
    "sandbox_external_retail_group1.py:test_case_ST009_004",
    "sandbox_external_retail_group1.py:test_case_ST009_005",
    "sandbox_external_retail_group1.py:test_case_ST010_001",
    "sandbox_external_retail_group1.py:test_case_ST011_002",
    "sandbox_external_retail_group1.py:test_case_ST011_006",
    "sandbox_external_retail_group1.py:test_case_ST011_007",
    "sandbox_external_retail_group1.py:test_case_ST011_009",
    "sandbox_external_retail_group1.py:test_case_ST012_003",
    "sandbox_external_retail_group1.py:test_case_ST013_001",
    "sandbox_external_retail_group1.py:test_case_ST013_003",
    "sandbox_external_retail_group1.py:test_case_ST013_004",
    "sandbox_external_retail_group1.py:test_case_ST013_005",
    "sandbox_external_retail_group1.py:test_case_ST013_006",
    "sandbox_external_retail_group1.py:test_case_ST013_007",
    "sandbox_external_retail_group1.py:test_case_ST013_008",
    "sandbox_external_retail_group1.py:test_case_ST014_002",
    "sandbox_external_retail_group1.py:test_case_ST014_005",
    "sandbox_external_retail_group1.py:test_case_ST015_001",
    "sandbox_external_retail_group1.py:test_case_ST016_004",
    "sandbox_external_retail_group1.py:test_case_ST016_006",
    "sandbox_external_retail_group1.py:test_case_ST017_001",
    "sandbox_external_retail_group1.py:test_case_ST017_002",
    "sandbox_external_retail_group1.py:test_case_ST017_003",
    "sandbox_external_retail_group1.py:test_case_ST017_005",
    "sandbox_external_retail_group1.py:test_case_ST018_002",
    "sandbox_external_retail_group1.py:test_case_ST018_003",
    "sandbox_external_retail_group1.py:test_case_ST018_005",
    "sandbox_external_retail_group1.py:test_case_ST025_001",
    "sandbox_external_retail_group1.py:test_case_ST025_002",
    "sandbox_external_retail_group1.py:test_case_ST025_003",
    "sandbox_external_retail_group1.py:test_case_ST025_004",
    "sandbox_external_retail_group1.py:test_case_ST025_006",
    "sandbox_external_retail_group1.py:test_case_ST026_002",
    "sandbox_external_retail_group1.py:test_case_ST026_003",
    "sandbox_external_retail_group1.py:test_case_ST028_003",
    "sandbox_external_retail_group1.py:test_case_ST029_001",
    "sandbox_external_retail_group1.py:test_case_ST029_002",
    "sandbox_external_retail_group1.py:test_case_ST030_001",
    "sandbox_external_retail_group1.py:test_case_ST030_002",
    "sandbox_external_retail_group1.py:test_case_ST030_003",
    "sandbox_external_retail_group1.py:test_case_ST031_001",
    "sandbox_external_retail_group1.py:test_case_ST031_002",
    "external_booking_v1_group1.py:test_bmd_001",
    "external_booking_v1_group1.py:test_bmd_010",
    "external_booking_v1_group1.py:test_bmd_014",
    "external_booking_v1_group1.py:test_bmd_015",
    "external_booking_v1_group1.py:test_bmd_017",
    "external_booking_v1_group1.py:test_bmd_019",
    "external_booking_v1_group1.py:test_bmd_021",
    "external_booking_v1_group1.py:test_bmd_022",
    "external_booking_v1_group1.py:test_bmd_023",
    "external_booking_v1_group1.py:test_bmd_024",
    "external_booking_v1_group1.py:test_bpy_002",
    "external_booking_v1_group1.py:test_bpy_003",
    "external_booking_v1_group1.py:test_bpy_007",
    "external_booking_v1_group1.py:test_bpy_008",
    "external_booking_v1_group1.py:test_bpy_009",
    "external_booking_v1_group1.py:test_bpy_010",
    "external_booking_v1_group1.py:test_cbi_001",
    "external_booking_v1_group1.py:test_cbi_003",
    "external_booking_v1_group1.py:test_cbi_004",
    "external_booking_v1_group1.py:test_cbi_005",
    "external_booking_v1_group1.py:test_cbm_001",
    "external_booking_v1_group1.py:test_cbm_002",
    "external_booking_v1_group1.py:test_cbm_005",
    "external_booking_v1_group1.py:test_cbm_006",
    "external_booking_v1_group1.py:test_cbm_007",
    "external_booking_v1_group1.py:test_cbm_009",
    "external_booking_v1_group1.py:test_cbm_010",
    "external_booking_v1_group1.py:test_cbm_012",
    "external_booking_v1_group1.py:test_ccn_003",
    "external_booking_v1_group1.py:test_ccn_005",
    "external_booking_v1_group1.py:test_ccn_007",
    "external_booking_v1_group1.py:test_crf_001",
    "external_booking_v1_group1.py:test_crf_003",
    "external_booking_v1_group1.py:test_crf_005",
    "external_booking_v1_group1.py:test_crf_007",
    "external_booking_v1_group1.py:test_crf_008",
    "external_booking_v1_group1.py:test_crf_010",
    "external_booking_v1_group1.py:test_crf_011",
    "external_booking_v1_group1.py:test_crf_012",
    "external_booking_v1_group1.py:test_crf_015",
    "external_booking_v1_group1.py:test_crf_016",
    "external_booking_v1_group1.py:test_crf_017",
    "external_booking_v1_group1.py:test_crf_018",
    "external_booking_v1_group1.py:test_crf_021",
    "external_booking_v1_group1.py:test_gbi_001",
    "external_booking_v1_group1.py:test_gbm_001",
    "external_booking_v1_group1.py:test_gbm_002",
    "external_booking_v1_group1.py:test_gbm_003",
    "external_booking_v1_group1.py:test_gbm_005",
    "external_booking_v1_group1.py:test_gbm_006",
    "external_booking_v1_group1.py:test_gbm_008",
    "external_booking_v1_group1.py:test_gbm_009",
    "external_booking_v1_group1.py:test_gbm_010",
    "external_booking_v1_group1.py:test_gbm_011",
    "external_booking_v1_group1.py:test_gbm_012",
    "external_booking_v1_group1.py:test_gbm_014",
    "external_booking_v1_group1.py:test_gbm_015",
    "external_booking_v1_group1.py:test_gbm_016",
    "external_booking_v1_group1.py:test_gbm_017",
    "external_booking_v1_group1.py:test_gbm_018",
    "external_booking_v1_group1.py:test_gsr_001",
    "external_booking_v1_group1.py:test_gsr_003",
    "external_booking_v1_group1.py:test_gsr_004",
    "external_booking_v1_group1.py:test_hdr_001",
    "external_booking_v1_group1.py:test_hdr_002",
    "external_booking_v1_group1.py:test_hdr_004",
    "external_booking_v1_group1.py:test_hdr_006",
    "external_booking_v1_group1.py:test_hdr_007",
    "external_booking_v1_group1.py:test_hpv_001",
    "external_booking_v1_group1.py:test_hpv_002",
    "external_booking_v1_group1.py:test_hpv_003",
    "external_booking_v1_group1.py:test_hpv_005",
    "external_booking_v1_group1.py:test_hpv_006",
    "external_booking_v1_group1.py:test_hpv_008",
    "external_booking_v1_group1.py:test_psc_001",
    "external_booking_v1_group1.py:test_psc_003",
    "external_booking_v1_group1.py:test_psc_004",
    "external_booking_v1_group1.py:test_psc_005",
    "external_booking_v1_group1.py:test_psc_008",
    "external_booking_v1_group1.py:test_psc_009",
    "external_booking_v1_group1.py:test_psc_010",
    "external_booking_v1_group1.py:test_psc_011",
    "external_booking_v1_group1.py:test_psc_012",
    "external_booking_v1_group1.py:test_psc_016",
    "external_booking_v1_group1.py:test_psc_018",
    "external_booking_v1_group1.py:test_pss_003",
    "external_booking_v1_group1.py:test_pss_007",
    "external_booking_v1_group1.py:test_pss_008",
    "external_booking_v1_group1.py:test_pss_010",
    "external_booking_v1_group1_rubrics_yesno.py:test_bmd_004",
    "external_booking_v1_group1_rubrics_yesno.py:test_bmd_007",
    "external_booking_v1_group1_rubrics_yesno.py:test_bmd_008",
    "external_booking_v1_group1_rubrics_yesno.py:test_bmd_012",
    "external_booking_v1_group1_rubrics_yesno.py:test_bmd_020",
    "external_booking_v1_group1_rubrics_yesno.py:test_bpy_001",
    "external_booking_v1_group1_rubrics_yesno.py:test_cbm_004",
    "external_booking_v1_group1_rubrics_yesno.py:test_ccn_002",
    "external_booking_v1_group1_rubrics_yesno.py:test_crf_006",
    "external_booking_v1_group1_rubrics_yesno.py:test_crf_014",
    "external_booking_v1_group1_rubrics_yesno.py:test_gbi_002",
    "external_booking_v1_group1_rubrics_yesno.py:test_gbm_007",
    "external_booking_v1_group1_rubrics_yesno.py:test_gsr_002",
    "external_booking_v1_group1_rubrics_yesno.py:test_psc_007",
    "external_booking_v1_group1_rubrics_yesno.py:test_psc_014",
    "sandbox_auto_insurance_group1.py:test_bil_001",
    "sandbox_auto_insurance_group1.py:test_bil_002",
    "sandbox_auto_insurance_group1.py:test_bil_003",
    "sandbox_auto_insurance_group1.py:test_bil_004",
    "sandbox_auto_insurance_group1.py:test_bil_005",
    "sandbox_auto_insurance_group1.py:test_bil_006",
    "sandbox_auto_insurance_group1.py:test_bil_007",
    "sandbox_auto_insurance_group1.py:test_bil_008",
    "sandbox_auto_insurance_group1.py:test_bil_009",
    "sandbox_auto_insurance_group1.py:test_bil_010",
    "sandbox_auto_insurance_group1.py:test_bil_011",
    "sandbox_auto_insurance_group1.py:test_bil_013",
    "sandbox_auto_insurance_group1.py:test_bil_014",
    "sandbox_auto_insurance_group1.py:test_bil_015",
    "sandbox_auto_insurance_group1.py:test_bil_016",
    "sandbox_auto_insurance_group1.py:test_bil_017",
    "sandbox_auto_insurance_group1.py:test_bil_018",
    "sandbox_auto_insurance_group1.py:test_bil_019",
    "sandbox_auto_insurance_group1.py:test_bil_020",
    "sandbox_auto_insurance_group1.py:test_bil_021",
    "sandbox_auto_insurance_group1.py:test_bil_022",
    "sandbox_auto_insurance_group1.py:test_bil_023",
    "sandbox_auto_insurance_group1.py:test_bil_024",
    "sandbox_auto_insurance_group1.py:test_bil_025",
    "sandbox_auto_insurance_group1.py:test_bil_101",
    "sandbox_auto_insurance_group1.py:test_bil_119",
    "sandbox_auto_insurance_group1.py:test_doc_002",
    "sandbox_auto_insurance_group1.py:test_doc_004",
    "sandbox_auto_insurance_group1.py:test_doc_005",
    "sandbox_auto_insurance_group1.py:test_doc_006",
    "sandbox_auto_insurance_group1.py:test_doc_007",
    "sandbox_auto_insurance_group1.py:test_drv_001",
    "sandbox_auto_insurance_group1.py:test_drv_002",
    "sandbox_auto_insurance_group1.py:test_drv_003",
    "sandbox_auto_insurance_group1.py:test_drv_005",
    "sandbox_auto_insurance_group1.py:test_drv_006",
    "sandbox_auto_insurance_group1.py:test_drv_007",
    "sandbox_auto_insurance_group1.py:test_drv_008",
    "sandbox_auto_insurance_group1.py:test_drv_009",
    "sandbox_auto_insurance_group1.py:test_drv_010",
    "sandbox_auto_insurance_group1.py:test_drv_011",
    "sandbox_auto_insurance_group1.py:test_drv_012",
    "sandbox_auto_insurance_group1.py:test_drv_013",
    "sandbox_auto_insurance_group1.py:test_drv_014",
    "sandbox_auto_insurance_group1.py:test_drv_015",
    "sandbox_auto_insurance_group1.py:test_drv_019",
    "sandbox_auto_insurance_group1.py:test_drv_020",
    "sandbox_auto_insurance_group1.py:test_drv_021",
    "sandbox_auto_insurance_group1.py:test_drv_022",
    "sandbox_auto_insurance_group1.py:test_drv_023",
    "sandbox_auto_insurance_group1.py:test_drv_101",
    "sandbox_auto_insurance_group1.py:test_drv_106",
    "sandbox_auto_insurance_group1.py:test_fnol_001",
    "sandbox_auto_insurance_group1.py:test_fnol_006",
    "sandbox_auto_insurance_group1.py:test_fnol_011",
    "sandbox_auto_insurance_group1.py:test_fnol_013",
    "sandbox_auto_insurance_group1.py:test_fnol_018",
    "sandbox_auto_insurance_group1.py:test_fnol_020",
    "sandbox_auto_insurance_group1.py:test_fnol_021",
    "sandbox_auto_insurance_group1.py:test_fnol_023",
    "sandbox_auto_insurance_group1.py:test_ldr_001",
    "sandbox_auto_insurance_group1.py:test_ldr_006",
    "sandbox_auto_insurance_group1.py:test_ldr_008",
    "sandbox_auto_insurance_group1.py:test_ldr_009",
    "sandbox_auto_insurance_group1.py:test_ldr_010",
    "sandbox_auto_insurance_group1.py:test_lif_001",
    "sandbox_auto_insurance_group1.py:test_lif_002",
    "sandbox_auto_insurance_group1.py:test_lif_003",
    "sandbox_auto_insurance_group1.py:test_lif_004",
    "sandbox_auto_insurance_group1.py:test_lif_005",
    "sandbox_auto_insurance_group1.py:test_lif_006",
    "sandbox_auto_insurance_group1.py:test_lif_007",
    "sandbox_auto_insurance_group1.py:test_lif_011",
    "sandbox_auto_insurance_group1.py:test_lif_012",
    "sandbox_auto_insurance_group1.py:test_lif_013",
    "sandbox_auto_insurance_group1.py:test_lif_014",
    "sandbox_auto_insurance_group1.py:test_lif_015",
    "sandbox_auto_insurance_group1.py:test_lif_017",
    "sandbox_auto_insurance_group1.py:test_lif_019",
    "sandbox_auto_insurance_group1.py:test_lif_020",
    "sandbox_auto_insurance_group1.py:test_lif_112",
    "sandbox_auto_insurance_group1.py:test_mul_001",
    "sandbox_auto_insurance_group1.py:test_veh_001",
    "sandbox_auto_insurance_group1.py:test_veh_002",
    "sandbox_auto_insurance_group1.py:test_veh_003",
    "sandbox_auto_insurance_group1.py:test_veh_004",
    "sandbox_auto_insurance_group1.py:test_veh_005",
    "sandbox_auto_insurance_group1.py:test_veh_007",
    "sandbox_auto_insurance_group1.py:test_veh_008",
    "sandbox_auto_insurance_group1.py:test_veh_009",
    "sandbox_auto_insurance_group1.py:test_veh_011",
    "sandbox_auto_insurance_group1.py:test_veh_012",
    "sandbox_auto_insurance_group1.py:test_veh_013",
    "sandbox_auto_insurance_group1.py:test_veh_014",
    "sandbox_auto_insurance_group1.py:test_veh_015",
    "sandbox_auto_insurance_group1.py:test_veh_018",
    "sandbox_auto_insurance_group1.py:test_veh_019",
    "sandbox_auto_insurance_group1.py:test_veh_020",
    "sandbox_auto_insurance_group1.py:test_veh_021",
    "sandbox_auto_insurance_group1.py:test_veh_113",
    "sandbox_neobank_support_v1_group1.py:test_dc_001",
    "sandbox_neobank_support_v1_group1.py:test_dc_003",
    "sandbox_neobank_support_v1_group1.py:test_ei_003",
    "sandbox_neobank_support_v1_group1.py:test_ei_004",
    "sandbox_neobank_support_v1_group1.py:test_ei_005",
    "sandbox_neobank_support_v1_group1.py:test_ei_006",
    "sandbox_neobank_support_v1_group1.py:test_ei_007",
    "sandbox_neobank_support_v1_group1.py:test_ei_008",
    "sandbox_neobank_support_v1_group1.py:test_ei_009",
    "sandbox_neobank_support_v1_group1.py:test_ei_012",
    "sandbox_neobank_support_v1_group1.py:test_ei_014",
    "sandbox_neobank_support_v1_group1.py:test_ei_017",
    "sandbox_neobank_support_v1_group1.py:test_ei_019",
    "sandbox_neobank_support_v1_group1.py:test_ei_020",
    "sandbox_neobank_support_v1_group1.py:test_ei_023",
    "sandbox_neobank_support_v1_group1.py:test_ei_024",
    "sandbox_neobank_support_v1_group1.py:test_ei_027",
    "sandbox_neobank_support_v1_group1.py:test_ei_028",
    "sandbox_neobank_support_v1_group1.py:test_ei_029",
    "sandbox_neobank_support_v1_group1.py:test_ei_030",
    "sandbox_neobank_support_v1_group1.py:test_ei_031",
    "sandbox_neobank_support_v1_group1.py:test_ei_032",
    "sandbox_neobank_support_v1_group1.py:test_ei_033",
    "sandbox_neobank_support_v1_group1.py:test_ei_034",
    "sandbox_neobank_support_v1_group1.py:test_ei_035",
    "sandbox_neobank_support_v1_group1.py:test_ei_036",
    "sandbox_neobank_support_v1_group1.py:test_ei_039",
    "sandbox_neobank_support_v1_group1.py:test_ei_040",
    "sandbox_neobank_support_v1_group1.py:test_he_001",
    "sandbox_neobank_support_v1_group1.py:test_he_002",
    "sandbox_neobank_support_v1_group1.py:test_he_003",
    "sandbox_neobank_support_v1_group1.py:test_he_004",
    "sandbox_neobank_support_v1_group1.py:test_he_006",
    "sandbox_neobank_support_v1_group1.py:test_he_007",
    "sandbox_neobank_support_v1_group1.py:test_he_011",
    "sandbox_neobank_support_v1_group1.py:test_he_012",
    "sandbox_neobank_support_v1_group1.py:test_he_013",
    "sandbox_neobank_support_v1_group1.py:test_he_014",
    "sandbox_neobank_support_v1_group1.py:test_he_016",
    "sandbox_neobank_support_v1_group1.py:test_he_017",
    "sandbox_neobank_support_v1_group1.py:test_he_021",
    "sandbox_neobank_support_v1_group1.py:test_he_023",
    "sandbox_neobank_support_v1_group1.py:test_he_024",
    "sandbox_neobank_support_v1_group1.py:test_he_025",
    "sandbox_neobank_support_v1_group1.py:test_he_027",
    "sandbox_neobank_support_v1_group1.py:test_he_028",
    "sandbox_neobank_support_v1_group1.py:test_he_030",
    "sandbox_neobank_support_v1_group1.py:test_he_035",
    "sandbox_neobank_support_v1_group1.py:test_he_036",
    "sandbox_neobank_support_v1_group1.py:test_he_037",
    "sandbox_neobank_support_v1_group1.py:test_he_038",
    "sandbox_neobank_support_v1_group1.py:test_he_040",
    "sandbox_neobank_support_v1_group1.py:test_ie_001",
    "sandbox_neobank_support_v1_group1.py:test_ie_003",
    "sandbox_neobank_support_v1_group1.py:test_ie_006",
    "sandbox_neobank_support_v1_group1.py:test_ie_009",
    "sandbox_neobank_support_v1_group1.py:test_ie_010",
    "sandbox_neobank_support_v1_group1.py:test_ie_011",
    "sandbox_neobank_support_v1_group1.py:test_pi_001",
    "sandbox_neobank_support_v1_group1.py:test_pi_002",
    "sandbox_neobank_support_v1_group1.py:test_pi_003",
    "sandbox_neobank_support_v1_group1.py:test_pi_005",
    "sandbox_neobank_support_v1_group1.py:test_pi_006",
    "sandbox_neobank_support_v1_group1.py:test_pi_008",
    "sandbox_neobank_support_v1_group1.py:test_pi_009",
    "sandbox_neobank_support_v1_group1.py:test_pi_014",
    "sandbox_neobank_support_v1_group1.py:test_sa_001",
    "sandbox_neobank_support_v1_group1.py:test_sa_002",
    "sandbox_neobank_support_v1_group1.py:test_sa_003",
    "sandbox_neobank_support_v1_group1.py:test_sa_004",
    "sandbox_neobank_support_v1_group1.py:test_sa_006",
    "sandbox_neobank_support_v1_group1.py:test_sa_008",
    "sandbox_neobank_support_v1_group1.py:test_sa_009",
    "sandbox_neobank_support_v1_group1.py:test_sa_010",
    "sandbox_neobank_support_v1_group1.py:test_sa_011",
    "sandbox_neobank_support_v1_group1.py:test_sa_013",
    "sandbox_neobank_support_v1_group1.py:test_sa_014",
    "sandbox_neobank_support_v1_group1.py:test_sa_015",
    "sandbox_neobank_support_v1_group1.py:test_sa_016",
    "sandbox_neobank_support_v1_group1.py:test_sa_018",
    "sandbox_neobank_support_v1_group1.py:test_sa_019",
    "sandbox_neobank_support_v1_group1.py:test_sa_020",
    "sandbox_neobank_support_v1_group1.py:test_sa_022",
    "sandbox_neobank_support_v1_group1.py:test_sa_023",
    "sandbox_neobank_support_v1_group1.py:test_sa_024",
    "sandbox_neobank_support_v1_group1.py:test_sa_025",
    "sandbox_neobank_support_v1_group1.py:test_sa_027",
    "sandbox_neobank_support_v1_group1.py:test_sa_029",
    "sandbox_neobank_support_v1_group1.py:test_sl_002",
    "sandbox_neobank_support_v1_group1_rubrics_yesno.py:test_aa_001",
    "sandbox_neobank_support_v1_group1_rubrics_yesno.py:test_he_009",
    "sandbox_neobank_support_v1_group1_rubrics_yesno.py:test_he_010",
    "sandbox_neobank_support_v1_group1_rubrics_yesno.py:test_he_020",
    "sandbox_neobank_support_v1_group1_rubrics_yesno.py:test_he_031",
    "sandbox_neobank_support_v1_group1_rubrics_yesno.py:test_he_032",
    "sandbox_neobank_support_v1_group1_rubrics_yesno.py:test_he_033",
    "sandbox_neobank_support_v1_group1_rubrics_yesno.py:test_he_039",
    "sandbox_neobank_support_v1_group1_rubrics_yesno.py:test_pi_007",
    "sandbox_neobank_support_v1_group1_rubrics_yesno.py:test_pi_011",
    "sandbox_neobank_support_v1_group1_rubrics_yesno.py:test_sa_005",
    "sandbox_neobank_support_v1_group1_rubrics_yesno.py:test_sa_012",
    "sandbox_neobank_support_v1_group1_rubrics_yesno.py:test_sa_026",
    "sandbox_neobank_support_v1_group1_rubrics_yesno.py:test_sa_028",
    "sandbox_neobank_support_v1_group1_rubrics_yesno.py:test_sl_004",
    "sandbox_consulting_group1.py:test_csa_001",
    "sandbox_consulting_group1.py:test_csa_003",
    "sandbox_consulting_group1.py:test_csa_004",
    "sandbox_consulting_group1.py:test_csa_005",
    "sandbox_consulting_group1.py:test_csa_006",
    "sandbox_consulting_group1.py:test_csa_007",
    "sandbox_consulting_group1.py:test_csa_008",
    "sandbox_consulting_group1.py:test_csa_009",
    "sandbox_consulting_group1.py:test_csa_010",
    "sandbox_consulting_group1.py:test_csa_012",
    "sandbox_consulting_group1.py:test_csa_013",
    "sandbox_consulting_group1.py:test_csa_016",
    "sandbox_consulting_group1.py:test_csa_018",
    "sandbox_consulting_group1.py:test_doc_001",
    "sandbox_consulting_group1.py:test_doc_002",
    "sandbox_consulting_group1.py:test_doc_003",
    "sandbox_consulting_group1.py:test_doc_004",
    "sandbox_consulting_group1.py:test_doc_005",
    "sandbox_consulting_group1.py:test_doc_007",
    "sandbox_consulting_group1.py:test_doc_009",
    "sandbox_consulting_group1.py:test_doc_010",
    "sandbox_consulting_group1.py:test_doc_013",
    "sandbox_consulting_group1.py:test_doc_014",
    "sandbox_consulting_group1.py:test_doc_015",
    "sandbox_consulting_group1.py:test_doc_016",
    "sandbox_consulting_group1.py:test_exp_001",
    "sandbox_consulting_group1.py:test_exp_002",
    "sandbox_consulting_group1.py:test_exp_003",
    "sandbox_consulting_group1.py:test_exp_004",
    "sandbox_consulting_group1.py:test_exp_005",
    "sandbox_consulting_group1.py:test_exp_006",
    "sandbox_consulting_group1.py:test_exp_007",
    "sandbox_consulting_group1.py:test_exp_008",
    "sandbox_consulting_group1.py:test_exp_009",
    "sandbox_consulting_group1.py:test_exp_010",
    "sandbox_consulting_group1.py:test_exp_011",
    "sandbox_consulting_group1.py:test_exp_012",
    "sandbox_consulting_group1.py:test_exp_013",
    "sandbox_consulting_group1.py:test_exp_014",
    "sandbox_consulting_group1.py:test_exp_015",
    "sandbox_consulting_group1.py:test_exp_018",
    "sandbox_consulting_group1.py:test_hwa_001",
    "sandbox_consulting_group1.py:test_hwa_003",
    "sandbox_consulting_group1.py:test_hwa_004",
    "sandbox_consulting_group1.py:test_hwa_005",
    "sandbox_consulting_group1.py:test_hwa_006",
    "sandbox_consulting_group1.py:test_hwa_007",
    "sandbox_consulting_group1.py:test_hwa_008",
    "sandbox_consulting_group1.py:test_hwa_009",
    "sandbox_consulting_group1.py:test_hwa_011",
    "sandbox_consulting_group1.py:test_hwa_012",
    "sandbox_consulting_group1.py:test_hwa_013",
    "sandbox_consulting_group1.py:test_hwa_014",
    "sandbox_consulting_group1.py:test_hwa_015",
    "sandbox_consulting_group1.py:test_hwa_016",
    "sandbox_consulting_group1.py:test_hwa_019",
    "sandbox_consulting_group1.py:test_hwa_020",
    "sandbox_consulting_group1.py:test_onb_001",
    "sandbox_consulting_group1.py:test_onb_002",
    "sandbox_consulting_group1.py:test_onb_003",
    "sandbox_consulting_group1.py:test_onb_005",
    "sandbox_consulting_group1.py:test_onb_007",
    "sandbox_consulting_group1.py:test_onb_009",
    "sandbox_consulting_group1.py:test_onb_010",
    "sandbox_consulting_group1.py:test_onb_011",
    "sandbox_consulting_group1.py:test_onb_013",
    "sandbox_consulting_group1.py:test_onb_014",
    "sandbox_consulting_group1.py:test_swa_001",
    "sandbox_consulting_group1.py:test_swa_002",
    "sandbox_consulting_group1.py:test_swa_003",
    "sandbox_consulting_group1.py:test_swa_004",
    "sandbox_consulting_group1.py:test_swa_005",
    "sandbox_consulting_group1.py:test_swa_006",
    "sandbox_consulting_group1.py:test_swa_007",
    "sandbox_consulting_group1.py:test_swa_009",
    "sandbox_consulting_group1.py:test_swa_012",
    "sandbox_consulting_group1.py:test_swa_013",
    "sandbox_consulting_group1.py:test_swa_016",
    "sandbox_consulting_group1.py:test_swa_017",
    "sandbox_consulting_group1.py:test_swa_019",
    "sandbox_consulting_group1.py:test_swa_020",
    "sandbox_consulting_group1.py:test_trn_001",
    "sandbox_consulting_group1.py:test_trn_002",
    "sandbox_consulting_group1.py:test_trn_003",
    "sandbox_consulting_group1.py:test_trn_004",
    "sandbox_consulting_group1.py:test_trn_005",
    "sandbox_consulting_group1.py:test_trn_006",
    "sandbox_consulting_group1.py:test_trn_008",
    "sandbox_consulting_group1.py:test_trn_011",
    "sandbox_consulting_group1.py:test_trn_018",
    "sandbox_consulting_group1.py:test_trn_021",
    "sandbox_consulting_group1.py:test_trn_022",
    "sandbox_consulting_group1.py:test_trv_001",
    "sandbox_consulting_group1.py:test_trv_002",
    "sandbox_consulting_group1.py:test_trv_004",
    "sandbox_consulting_group1.py:test_trv_006",
    "sandbox_consulting_group1.py:test_trv_009",
    "sandbox_consulting_group1.py:test_trv_010",
    "sandbox_consulting_group1.py:test_trv_011",
    "sandbox_consulting_group1.py:test_trv_012",
    "sandbox_consulting_group1.py:test_trv_013",
)


class SubmittedToolCall(BaseModel):
    """Represent one provider tool call within an assistant turn.

    Args:
        name (`str`):
            Tool name supplied by the provider.
        arguments (`dict`, *optional*):
            Parsed tool arguments.
        call_id (`str`):
            Provider call identifier.
        parse_error (`str`, *optional*):
            Provider parse error that prevents execution.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str
    parse_error: str | None = Field(
        default=None,
        description="Native provider parse error; such calls are never executed.",
    )


class ToolCallResult(BaseModel):
    """Represent one model-visible result in a parallel tool batch.

    Args:
        name (`str`):
            Tool name.
        call_id (`str`):
            Provider call identifier.
        content (`str`):
            Model-visible tool response.
        tool_error (`str`, *optional*):
            Stable public error classification.
        direct_response (`str`, *optional*):
            Native direct-response rendering.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    call_id: str
    content: str
    tool_error: Literal["tool_not_found", "invalid_args", "execution_error"] | None = (
        None
    )
    direct_response: str | None = None


class CallToolAction(Action):
    """Call one tool or one native parallel tool batch.

    Args:
        tool_name (`str`, *optional*):
            Single tool name.
        arguments (`dict`, *optional*):
            Single-call arguments.
        call_id (`str`, *optional*):
            Single provider call identifier.
        parallel_tool_calls (`list[SubmittedToolCall]`, *optional*):
            Ordered native provider batch.
    """

    type: Literal["call_tool"] = "call_tool"
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = Field(
        default=None,
        description="Provider tool-call identifier, when the model supplied one.",
    )
    parallel_tool_calls: list[SubmittedToolCall] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_single_or_parallel(self) -> "CallToolAction":
        """Require exactly one valid single-call or parallel-call representation."""
        has_single = self.tool_name is not None
        has_parallel = bool(self.parallel_tool_calls)
        if has_single == has_parallel:
            raise ValueError(
                "supply exactly one tool_name or a non-empty parallel_tool_calls batch"
            )
        if has_parallel and (self.arguments or self.call_id is not None):
            raise ValueError(
                "batch calls carry arguments and call IDs on each child call"
            )
        call_ids = [call.call_id for call in self.parallel_tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("parallel tool call IDs must be unique")
        return self


class SubmitMessageAction(Action):
    """Submit assistant-visible text or a terminal provider turn.

    Args:
        content (`str`, *optional*):
            Assistant text visible to the user and grader.
        terminal_tool_calls (`list[SubmittedToolCall]`, *optional*):
            Provider calls recorded but not executed after terminal ordering.
        tool_calls_before_content (`bool`, *optional*, defaults to `False`):
            Whether the provider emitted terminal calls before visible text.
    """

    type: Literal["submit_message"] = "submit_message"
    content: str | None = None
    terminal_tool_calls: list[SubmittedToolCall] = Field(default_factory=list)
    tool_calls_before_content: bool = False

    @model_validator(mode="after")
    def require_visible_turn(self) -> "SubmitMessageAction":
        """Require visible text or at least one terminal provider call."""
        if self.content is None and not self.terminal_tool_calls:
            raise ValueError(
                "content is required unless terminal_tool_calls are supplied"
            )
        return self


class _FinishAction(Action):
    """Trusted harness finish action; intentionally absent from public schemas."""

    type: Literal["_finish"] = "_finish"
    reason: str = "harness"


_PublicAction = Annotated[
    ListToolsAction | CallToolAction | SubmitMessageAction,
    Field(discriminator="type"),
]
_WireAction = Annotated[
    ListToolsAction | CallToolAction | SubmitMessageAction | _FinishAction,
    Field(discriminator="type"),
]
_public_action_adapter = TypeAdapter(_PublicAction)
_wire_action_adapter = TypeAdapter(_WireAction)


class ThinkingBoxAction(Action):
    """Expose the discriminated public action union used on the wire."""

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Action:  # type: ignore[override]
        """Validate a payload into its concrete ThinkingBox action.

        Args:
            obj (`object`):
                Python payload to validate.
            kwargs (`dict`, *optional*):
                Additional Pydantic validation arguments.

        Returns:
            [`~openenv.core.Action`]:
                Concrete list-tools, tool-call, message, or trusted finish action.
        """
        return _wire_action_adapter.validate_python(obj, **kwargs)

    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """Return the public schema without the trusted harness finish action.

        Args:
            kwargs (`dict`, *optional*):
                Additional Pydantic schema arguments.

        Returns:
            `dict`:
                JSON schema for model-visible actions.
        """
        return _public_action_adapter.json_schema(**kwargs)


class ThinkingBoxExecutionProvenance(BaseModel):
    """Describe the public, credential-free server execution identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thinkingbox_revision: str
    thinkingbox_source_sha256: str
    thinkingbox_source_type: str
    data_release: str
    data_revision: str
    config_sha256: str | None
    data_bundle_sha256: str
    manifest_path: str
    manifest_sha256: str
    manifest_uids_sha256: str
    task_count: int = Field(gt=0)


class ThinkingBoxObservation(Observation):
    """Represent the privacy-reviewed model-visible episode observation.

    Reset observations expose the task, instructions, permitted tools, and
    visible message history. Step observations add tool or simulated-user
    output, while terminal observations add binary grading status and public
    provenance without private benchmark state.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["reset", "tools", "tool", "tool_batch", "user", "terminal", "error"]
    task_uid: str | None = None
    task: str | None = None
    system_instructions: str | None = None
    bot_instructions: str | None = None
    tools: list[Tool] | None = None
    messages: list[dict[str, Any]] | None = None
    tool_name: str | None = None
    call_id: str | None = None
    tool_result: str | None = None
    tool_error: Literal["tool_not_found", "invalid_args", "execution_error"] | None = (
        None
    )
    direct_response: str | None = None
    tool_results: list[ToolCallResult] | None = None
    user_message: str | None = None
    response: str | None = None
    finish_reason: str | None = None
    reward_type: Literal["pass", "fail", "system_error"] | None = None
    system_error: bool = False
    test_summary: dict[str, Any] | None = None
    error: str | None = None
    steps_taken: int = 0

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize the observation without absent optional fields.

        Args:
            kwargs (`dict`, *optional*):
                Additional Pydantic serialization arguments.

        Returns:
            `dict`:
                JSON-compatible observation payload.
        """
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


class ThinkingBoxState(State):
    """Represent non-sensitive OpenEnv lifecycle state.

    Args:
        task_uid (`str`, *optional*):
            Active canonical task UID.
        status (`str`, *optional*, defaults to `"idle"`):
            Current server lifecycle state.
        system_error (`bool`, *optional*, defaults to `False`):
            Whether an infrastructure failure was latched.
    """

    task_uid: str | None = None
    status: Literal["idle", "active", "finalizing", "done", "closed", "error"] = "idle"
    system_error: bool = False
