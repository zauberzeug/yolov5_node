import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

import train_det
from app_code.training_checkpoint import atomic_save, discard_unfinished_epochs, restore_training_state, save_best
from app_code.yolov5_trainer import Yolov5TrainerLogic


def test_atomic_save_preserves_checkpoint_on_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'last.pt'
    atomic_save({'epoch': 4}, path)

    def failed_write(_state: Any, handle: Any) -> None:
        handle.write(b'incomplete')
        raise OSError('interrupted')

    monkeypatch.setattr(torch, 'save', failed_write)
    with pytest.raises(OSError, match='interrupted'):
        atomic_save({'epoch': 5}, path)
    assert torch.load(path)['epoch'] == 4


def full_checkpoint(epoch: int) -> dict[str, Any]:
    return {'epoch': epoch, 'best_fitness': 0.5, 'model': torch.nn.Linear(1, 1), 'ema': torch.nn.Linear(1, 1),
            'updates': 3, 'optimizer': {'state': {}}, 'opt': {}, 'date': 'today'}


def test_best_keeps_only_ema_weights_and_links_the_published_epoch(tmp_path: Path) -> None:
    checkpoint = full_checkpoint(4)
    save_best(checkpoint, tmp_path, {'cat': {'tp': 2}})

    best = torch.load(tmp_path / 'best.pt', weights_only=False)
    assert best['optimizer'] is None and best['ema'] is None
    torch.testing.assert_close(best['model'].weight, checkpoint['ema'].weight)
    assert (tmp_path / 'epoch4.pt').stat().st_ino == (tmp_path / 'best.pt').stat().st_ino
    assert (tmp_path / 'epoch4.json').read_text() == '{"cat": {"tp": 2}}'


def test_restart_discards_best_of_an_epoch_without_last(tmp_path: Path) -> None:
    weights = tmp_path / 'result/weights'
    weights.mkdir(parents=True)
    save_best(full_checkpoint(4), weights, {'cat': {'tp': 2}})
    save_best(full_checkpoint(5), weights, {'cat': {'tp': 3}})

    discard_unfinished_epochs(weights, last_epoch=4)

    assert sorted(path.name for path in weights.iterdir()) == ['epoch4.json', 'epoch4.pt']


def test_resume_restores_optimizer_ema_schedule_scaler_and_patience(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    scaler = torch.amp.GradScaler('cpu', growth_interval=1)
    ema = train_det.ModelEMA(model)
    stopper = train_det.EarlyStopping(patience=3)

    def step(m: Any, opt: Any, sched: Any, scale: Any, average: Any) -> None:
        opt.zero_grad()
        scale.scale(m(torch.ones(1, 1)).square().sum()).backward()
        scale.step(opt)
        scale.update()
        sched.step()
        average.update(m)

    for epoch, score in enumerate([0.8, 0.7, 0.6]):
        step(model, optimizer, scheduler, scaler, ema)
        stopper(epoch, score)
    checkpoint = {'epoch': 2, 'best_fitness': 0.8, 'model': deepcopy(model), 'ema': deepcopy(ema.ema),
                  'updates': ema.updates, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                  'scaler': scaler.state_dict(),
                  'early_stopping': {'best_epoch': stopper.best_epoch, 'best_fitness': stopper.best_fitness,
                                     'possible_stop': stopper.possible_stop}}
    atomic_save(checkpoint, tmp_path / 'last.pt')
    saved = torch.load(tmp_path / 'last.pt', weights_only=False)
    resumed = deepcopy(saved['model'])
    new_optimizer = torch.optim.Adam(resumed.parameters(), lr=0.01)
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1, gamma=0.8)
    new_scaler = torch.amp.GradScaler('cpu', growth_interval=1)
    new_ema = train_det.ModelEMA(resumed)
    new_stopper = train_det.EarlyStopping(patience=3)
    best, start, epochs = train_det.smart_resume(saved, new_optimizer, new_ema, epochs=10)
    restore_training_state(saved, new_scheduler, new_scaler, new_stopper)

    assert (best, start, epochs) == (0.8, 3, 10)
    assert new_stopper(3, 0.5)
    assert new_scaler.state_dict() == scaler.state_dict()
    assert new_scheduler.state_dict() == scheduler.state_dict()
    step(model, optimizer, scheduler, scaler, ema)
    step(resumed, new_optimizer, new_scheduler, new_scaler, new_ema)
    for actual, expected in zip(resumed.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(new_ema.ema.parameters(), ema.ema.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert new_ema.updates == ema.updates


def test_main_resumes_with_original_batch_size_and_without_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / 'result/weights'
    weights.mkdir(parents=True)
    checkpoint = weights / 'last.pt'
    checkpoint.touch()
    monkeypatch.setattr('sys.argv', ['train_det.py'])
    options = train_det.parse_opt()
    options.clear = True
    options.batch_size = 4
    options.save_dir = str(weights.parent)
    train_det.yaml_save(str(weights.parent / 'opt.yaml'), vars(options))
    options.resume = str(checkpoint)
    options.batch_size = 32
    train = MagicMock()
    monkeypatch.setattr(train_det, 'train', train)
    monkeypatch.setattr(train_det, 'select_device', lambda *_a, **_kw: torch.device('cpu'))
    monkeypatch.setattr(train_det, 'check_comet_resume', lambda _: False)

    train_det.main(options)

    resumed = train.call_args.args[1]
    assert resumed.resume is True
    assert resumed.clear is False
    assert resumed.batch_size == 4
    assert resumed.weights == str(checkpoint)


def test_node_uses_last_checkpoint_without_batch_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / 'result/weights'
    (weights / 'published').mkdir(parents=True)
    (weights / 'last.pt').touch()
    (weights / 'published/latest.pt').touch()
    (weights.parent / 'opt.yaml').write_text('batch_size: 4\n')
    (tmp_path / 'hyp.yaml').write_text('epochs: 10\n')
    logic = Yolov5TrainerLogic()
    logic._training = SimpleNamespace(training_folder=str(tmp_path), training_folder_path=tmp_path, hyperparameters={})
    logic._executor = MagicMock(start=AsyncMock())
    probe = AsyncMock(side_effect=AssertionError('must not probe'))
    monkeypatch.setattr('app_code.yolov5_trainer.batch_size_calculation.calc', probe)
    assert logic._can_resume()

    asyncio.run(logic._resume())

    assert logic.executor.start.call_args.args[0] == f'python /app/train_det.py --resume {weights / "last.pt"}'
    assert logic.training.hyperparameters['batch_size'] == 4
    probe.assert_not_called()


def test_completed_checkpoint_does_not_start_another_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('sys.argv', ['train_det.py'])
    options = train_det.parse_opt()
    options.resume = True
    options.clear = False
    options.save_dir = str(tmp_path / 'result')
    options.weights = str(tmp_path / 'last.pt')
    atomic_save({'epoch': 3, 'stopped_early': True}, Path(options.weights))
    monkeypatch.setattr(train_det, 'Loggers', lambda *_: SimpleNamespace(remote_dataset=None))
    monkeypatch.setattr(train_det, 'methods', lambda _: [])
    monkeypatch.setattr(train_det, 'check_dataset', lambda _: {'train': 'train', 'val': 'val', 'nc': 1, 'names': ['cat']})
    model = MagicMock(side_effect=AssertionError('completed training must not build another model'))
    monkeypatch.setattr(train_det, 'Model', model)
    assert train_det.train({}, options, torch.device('cpu'), train_det.Callbacks()) == (0.0,) * 7
    model.assert_not_called()
