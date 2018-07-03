import unittest
from cossmo import validation_monitor


class TestValidationMonitor(unittest.TestCase):
    def test_validation_monitor_maximize(self):
        vm = validation_monitor.ValidationMonitor(True, 3)

        vm(1, 1)
        vm(2, 2)
        vm(3, 3)

        self.assertFalse(vm.should_stop())

        vm(3, 1)
        vm(2, 2)
        vm(1, 3)

        self.assertTrue(vm.should_stop())

        vm(5, 1)
        vm(4, 2)
        vm(5, 3)

        self.assertFalse(vm.should_stop())

        vm(5, 1)
        vm(5, 2)
        vm(6, 3)

        self.assertFalse(vm.should_stop())

    def test_validation_monitor_minimize(self):
        vm = validation_monitor.ValidationMonitor(False, 3)

        vm(3, 1)
        vm(2, 2)
        vm(1, 3)

        self.assertFalse(vm.should_stop())

        vm(1, 1)
        vm(2, 2)
        vm(3, 3)

        self.assertTrue(vm.should_stop())

        vm(5, 1)
        vm(4, 2)
        vm(5, 3)

        self.assertFalse(vm.should_stop())

        vm(5, 1)
        vm(5, 2)
        vm(4, 3)

        self.assertFalse(vm.should_stop())
