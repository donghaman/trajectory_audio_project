import pytest
from unittest.mock import patch

from audio_engine import AudioEngine


def test_mapping_defaults_are_distinct():
    engine = AudioEngine()
    engine.validate_mapping()
    assert engine.mapping == {'FL': 1, 'FR': 2, 'RL': 3, 'RR': 4}


def test_mapping_rejects_duplicate_channel():
    engine = AudioEngine()
    with pytest.raises(ValueError, match='서로 다른'):
        engine.validate_mapping({'FL': 1, 'FR': 1, 'RL': 3, 'RR': 4})


def test_mapping_rejects_out_of_range_channel():
    engine = AudioEngine()
    with pytest.raises(ValueError, match='1부터 64'):
        engine.validate_mapping({'FL': 0, 'FR': 2, 'RL': 3, 'RR': 4})


def test_validate_device_rejects_less_than_four_channels():
    engine = AudioEngine()
    with patch('audio_engine.sd.query_devices', return_value={'name': 'Stereo Device', 'max_output_channels': 2}):
        with pytest.raises(ValueError, match='최대 2채널'):
            engine.validate_device(0, 4)


def test_validate_device_accepts_four_channels():
    engine = AudioEngine()
    with patch('audio_engine.sd.query_devices', return_value={'name': 'USB 4-Channel', 'max_output_channels': 4}):
        engine.validate_device(0, 4)


def test_validate_device_accepts_two_channels_for_earphone():
    engine = AudioEngine()
    with patch('audio_engine.sd.query_devices', return_value={'name': 'Stereo Headphones', 'max_output_channels': 2}):
        engine.validate_device(0, 2)


def test_validate_device_rejects_one_channel_for_earphone():
    engine = AudioEngine()
    with patch('audio_engine.sd.query_devices', return_value={'name': 'Mono Device', 'max_output_channels': 1}):
        with pytest.raises(ValueError, match='최대 1채널'):
            engine.validate_device(0, 2)
