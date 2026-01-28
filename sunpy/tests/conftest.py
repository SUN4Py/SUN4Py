"""
SUNPy A Python Library for solving SU(N) Heisenberg models
Copyright (C) 2026  Samuel Gozel, GNU GPLv3

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pytest
import os
import shutil


test_durations = {}

@pytest.fixture(scope="session", autouse=True)
def run_tests():
    """
    
    """
    test_dirname = os.path.dirname(os.path.abspath(__file__))
    temp_dirname = os.path.join(test_dirname, 'temp')
    if not os.path.exists(temp_dirname):
        os.mkdir(temp_dirname)
    
    yield # Run all tests
    
    if os.path.exists(temp_dirname):
        shutil.rmtree(temp_dirname)
    return

'''
# currently not running on my old system
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to capture test execution time for each test.
    This runs for every test phase (setup, call, teardown).
    """
    outcome = yield
    report = outcome.get_result()
    
    # Only capture the actual test execution time (not setup/teardown)
    if report.when == 'call':
        test_name = item.nodeid
        duration = call.duration
        test_durations[test_name] = duration


def pytest_sessionfinish(session, exitstatus):
    """
    Hook called after all tests finish.
    Print a summary of all test execution times.
    """
    print("\n" + "="*70)
    print("TEST EXECUTION TIME REPORT")
    print("="*70)
    
    if test_durations:
        # Sort tests by duration (slowest first)
        sorted_tests = sorted(test_durations.items(), key=lambda x: x[1], reverse=True)
        
        for test_name, duration in sorted_tests:
            print(f"{duration:>8.4f}s - {test_name}")
        
        # Calculate and display statistics
        total_time = sum(test_durations.values())
        avg_time = total_time / len(test_durations)
        slowest = max(test_durations.values())
        fastest = min(test_durations.values())
        
        print("="*70)
        print(f"Total tests:     {len(test_durations)}")
        print(f"Total time:      {total_time:.4f}s")
        print(f"Average time:    {avg_time:.4f}s")
        print(f"Slowest test:    {slowest:.4f}s")
        print(f"Fastest test:    {fastest:.4f}s")
        print("="*70)
    else:
        print("No tests were executed.")
        print("="*70)
'''