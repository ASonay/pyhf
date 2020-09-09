import pytest
import pyhf
import numpy as np


def calculate_CLs(bkgonly_json, signal_patch_json):
    """
    Calculate the observed CLs and the expected CLs band from a background only
    and signal patch.

    Args:
        bkgonly_json: The JSON for the background only model
        signal_patch_json: The JSON Patch for the signal model

    Returns:
        CLs_obs: The observed CLs value
        CLs_exp: List of the expected CLs value band
    """
    workspace = pyhf.workspace.Workspace(bkgonly_json)
    model = workspace.model(
        measurement_name=None,
        patches=[signal_patch_json],
        modifier_settings={
            'normsys': {'interpcode': 'code4'},
            'histosys': {'interpcode': 'code4p'},
        },
    )
    result = pyhf.infer.hypotest(
        1.0, workspace.data(model), model, qtilde=True, return_expected_set=True
    )
    return result[0].tolist(), result[-1]


def test_sbottom_regionA_1300_205_60(
    sbottom_likelihoods_download, get_json_from_tarfile
):
    sbottom_regionA_bkgonly_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionA/BkgOnly.json"
    )
    sbottom_regionA_1300_205_60_patch_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionA/patch.sbottom_1300_205_60.json"
    )
    CLs_obs, CLs_exp = calculate_CLs(
        sbottom_regionA_bkgonly_json, sbottom_regionA_1300_205_60_patch_json
    )
    assert CLs_obs == pytest.approx(0.2444418, rel=1e-5)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    np.array(0.09023417),
                    np.array(0.19379784),
                    np.array(0.38434083),
                    np.array(0.65579069),
                    np.array(0.89104895),
                ]
            ),
            rtol=1e-5,
        )
    )


def test_sbottom_regionA_1400_950_60(
    sbottom_likelihoods_download, get_json_from_tarfile
):
    sbottom_regionA_bkgonly_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionA/BkgOnly.json"
    )
    sbottom_regionA_1400_950_60_patch_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionA/patch.sbottom_1400_950_60.json"
    )
    CLs_obs, CLs_exp = calculate_CLs(
        sbottom_regionA_bkgonly_json, sbottom_regionA_1400_950_60_patch_json
    )
    assert CLs_obs == pytest.approx(0.0213732, rel=1e-5)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    np.array(0.00264459),
                    np.array(0.01397628),
                    np.array(0.06497158),
                    np.array(0.23644149),
                    np.array(0.57448001),
                ],
            ),
            rtol=1e-5,
        )
    )


def test_sbottom_regionA_1500_850_60(
    sbottom_likelihoods_download, get_json_from_tarfile
):
    sbottom_regionA_bkgonly_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionA/BkgOnly.json"
    )
    sbottom_regionA_1500_850_60_patch_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionA/patch.sbottom_1500_850_60.json"
    )
    CLs_obs, CLs_exp = calculate_CLs(
        sbottom_regionA_bkgonly_json, sbottom_regionA_1500_850_60_patch_json
    )
    assert CLs_obs == pytest.approx(0.04536628, rel=1e-5)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    np.array(0.00598365),
                    np.array(0.02610002),
                    np.array(0.10093043),
                    np.array(0.31018133),
                    np.array(0.65535131),
                ]
            ),
            rtol=1e-5,
        )
    )


def test_sbottom_regionB_1400_550_60(
    sbottom_likelihoods_download, get_json_from_tarfile
):
    sbottom_regionB_bkgonly_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionB/BkgOnly.json"
    )
    sbottom_regionB_1400_550_60_patch_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionB/patch.sbottom_1400_550_60.json"
    )
    CLs_obs, CLs_exp = calculate_CLs(
        sbottom_regionB_bkgonly_json, sbottom_regionB_1400_550_60_patch_json
    )
    assert CLs_obs == pytest.approx(0.9944320, rel=1e-5)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    np.array(0.9939882),
                    np.array(0.99613164),
                    np.array(0.99797367),
                    np.array(0.99926868),
                    np.array(0.99985933),
                ]
            ),
            rtol=1e-5,
        )
    )


def test_sbottom_regionC_1600_850_60(
    sbottom_likelihoods_download, get_json_from_tarfile
):
    sbottom_regionC_bkgonly_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionC/BkgOnly.json"
    )
    sbottom_regionC_1600_850_60_patch_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionC/patch.sbottom_1600_850_60.json"
    )
    CLs_obs, CLs_exp = calculate_CLs(
        sbottom_regionC_bkgonly_json, sbottom_regionC_1600_850_60_patch_json
    )
    assert CLs_obs == pytest.approx(0.711023707425625, rel=1e-5)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    np.array(0.29555621),
                    np.array(0.44469567),
                    np.array(0.63715336),
                    np.array(0.83361842),
                    np.array(0.9585912),
                ]
            ),
            rtol=1e-5,
        )
    )
