from my_utils.train_and_eval._score_monitor import *

def test_init():
    threshold = 4
    monitor = ScoreMonitor(threshold)
    assert threshold == monitor.threshold
    assert True == monitor.go_up
    monitor = ScoreMonitor(go_up=False)
    assert False == monitor.go_up

def test_update_best():
    monitor = ScoreMonitor()
    assert True == monitor.go_up
    monitor.update_best(45)
    assert 0 == monitor.stop_count
    monitor.update_best(50)
    assert 0 == monitor.stop_count
    monitor.update_best(45)
    assert 1 == monitor.stop_count
    monitor.update_best(40)
    assert 2 == monitor.stop_count
    monitor.update_best(47)
    assert 3 == monitor.stop_count
    monitor.update_best(51)
    assert 0 == monitor.stop_count

    monitor = ScoreMonitor(go_up=False)
    assert False == monitor.go_up
    monitor.update_best(45)
    assert 0 == monitor.stop_count
    monitor.update_best(40)
    assert 0 == monitor.stop_count
    monitor.update_best(42)
    assert 1 == monitor.stop_count
    monitor.update_best(46)
    assert 2 == monitor.stop_count
    monitor.update_best(45)
    assert 3 == monitor.stop_count
    monitor.update_best(20)
    assert 0 == monitor.stop_count

def test_check_stop():
    monitor = ScoreMonitor()
    assert True == monitor.go_up
    assert 1 == monitor.threshold
    monitor.update_best(45)
    assert False == monitor.check_stop()
    monitor.update_best(20)
    assert True == monitor.check_stop()

    threshold = 3
    go_up = False
    monitor = ScoreMonitor(threshold=threshold, go_up=go_up)
    assert go_up == monitor.go_up
    assert threshold == monitor.threshold
    monitor.update_best(45)
    monitor.update_best(20)
    monitor.update_best(30)
    monitor.update_best(40)
    assert False == monitor.check_stop()
    monitor.update_best(45)
    assert True == monitor.check_stop()
