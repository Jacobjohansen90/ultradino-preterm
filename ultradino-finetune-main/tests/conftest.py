from pathlib import Path


import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--ultradino-finetune-test-data',
        metavar='BASE_PATH',
        type=Path,
        default='/data/proto/CommonCode/ultradino-finetune/test-data',
    )


@pytest.fixture
def test_data_base_path(request):
    yield request.config.getoption('ultradino_finetune_test_data')
