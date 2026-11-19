"""Real admitted samples/checkpoints: rejection and cross-process evidence."""
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from banksim.data import DataError, sha256_file
from banksim.forecast_checkpoint import load_checkpoint
from banksim.forecast_stage import setup
from banksim.forecast_train_cli import saved_catalogs
from banksim.forecast_training import RunConfig, train_model
from banksim.retrieval_io import read_json

ROOT = Path('artifacts/stage4/resume_validation/final')


@pytest.fixture(scope='module')
def real_run():
    engines = setup()
    directory = ROOT/'comparison/O/A'
    return engines,directory,saved_catalogs(engines['O'],directory)


@pytest.mark.parametrize('field,value', [('seed',43),('variant','no_retrieval'),('method','all'),
                                        ('max_epochs',4),('batch_size',32)])
def test_resume_rejects_configuration_change(real_run,field,value):
    _,directory,parts = real_run
    before = sha256_file(directory/'checkpoint_manifest.json')
    with pytest.raises(DataError) as exc:
        train_model(parts['train'],parts['validation'],directory,replace(RunConfig(),**{field:value}),
                    directory/'cache',resume=True)
    assert exc.value.code == 'CONFIG_MISMATCH'
    assert sha256_file(directory/'checkpoint_manifest.json') == before


def test_experiment_and_smoke_promotion_rejected(real_run):
    engines,directory,_ = real_run
    for engine,kind in ((engines['C'],'smoke'),(engines['O'],'full')):
        with pytest.raises(DataError) as exc:
            load_checkpoint(directory,engine,kind=kind,name='last')
        assert exc.value.code == 'CONFIG_MISMATCH'


def test_changed_sample_selection_is_rejected(real_run):
    _,directory,parts = real_run
    smaller = parts['train'].subset(parts['train'].customers[:2],list(range(59,71)))
    with pytest.raises(DataError) as exc:
        train_model(smaller,parts['validation'],directory,RunConfig(),directory/'cache',resume=True)
    assert exc.value.code == 'CONFIG_MISMATCH'


def test_code_data_preprocessing_versions_and_old_smoke_rejected(real_run,monkeypatch):
    import banksim.forecast_checkpoint as checkpoints
    engines,directory,_ = real_run
    engine = engines['O']
    for key,value in (('data_hash','different'),('cluster_version','different'),('implementation_version','different')):
        with monkeypatch.context() as patch:
            patch.setitem(engine.versions,key,value)
            with pytest.raises(DataError) as exc:
                load_checkpoint(directory,engine,kind='smoke',name='last')
            assert exc.value.code == 'CONFIG_MISMATCH'
    with monkeypatch.context() as patch:
        patch.setitem(engine.bundle.amount_scale,'a',999.)
        with pytest.raises(DataError,match='version differs'):
            load_checkpoint(directory,engine,kind='smoke',name='last')
    with monkeypatch.context() as patch:
        patch.setattr(checkpoints,'model_code_version',lambda:'changed-code')
        with pytest.raises(DataError,match='version differs'):
            load_checkpoint(directory,engine,kind='smoke',name='last')
    with pytest.raises(DataError) as exc:
        load_checkpoint('artifacts/stage4/smoke/O/retrieval',engine,kind='smoke')
    assert exc.value.code == 'CONFIG_MISMATCH'  # Preserve original metadata, never rewrite it.


def test_manifest_tamper_rejected(real_run,tmp_path):
    engines,directory,_ = real_run
    destination = tmp_path/'r'
    shutil.copytree(directory,destination,ignore=shutil.ignore_patterns('cache','progress.jsonl'))
    with (destination/'train_selection.parquet').open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(DataError,match='manifest changed'):
        load_checkpoint(destination,engines['O'],kind='smoke',name='last')


def test_new_never_overwrites_existing_run(real_run):
    _,directory,parts = real_run
    before = sha256_file(directory/'checkpoint_manifest.json')
    with pytest.raises(DataError,match='explicit resume'):
        train_model(parts['train'],parts['validation'],directory,RunConfig(),directory/'cache')
    assert sha256_file(directory/'checkpoint_manifest.json') == before


def test_actual_independent_process_comparison_and_early_stop_counter():
    report = read_json(ROOT/'comparison.json')
    assert report['status'] == 'PASS' and report['plan']['tolerance']['absolute'] == 0
    commands = report['processes']
    assert all(c['exit_code'] == c['expected_exit'] for c in commands)
    for experiment in ('O','C'):
        by_name = {c['name']:c for c in commands}
        assert by_name[experiment+'_B_stop']['pid'] != by_name[experiment+'_B_resume']['pid']
        assert by_name[experiment+'_C_stop']['pid'] != by_name[experiment+'_C_resume']['pid']
        assert all(r['all_last_state_numeric_equal'] and r['max_absolute_difference'] == 0
                   for r in report['results'][experiment]['comparisons_to_A'].values())
    pending = read_json(ROOT/'validation-interrupted-state.json')
    assert pending['bad_epochs'] == 1 and pending['phase'] == 'validating'
    assert report['results']['O']['comparisons_to_A']['D']['bad_epochs'] == 2
    assert len(report['results']['O']['curve']) == 3
