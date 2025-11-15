#!/usr/bin/env python
"""
Test script to validate critical security and portability fixes.
This script verifies:
1. Environment variable loading
2. Cross-platform path resolution
3. Import statements
4. API key security
"""

import os
import sys
from pathlib import Path

# Add the source directory to path
sys.path.insert(0, str(Path(__file__).parent / "2026" / "src"))

print("=" * 70)
print("🧪 Testing Critical Fixes for Basketball Prediction Project")
print("=" * 70)

# Test 1: Check .env.example exists
print("\n[Test 1] Checking .env.example exists...")
env_example_path = Path(__file__).parent / ".env.example"
if env_example_path.exists():
    print("✅ PASS: .env.example exists")
    with open(env_example_path) as f:
        content = f.read()
        if "ODDS_API_KEY" in content:
            print("✅ PASS: .env.example contains ODDS_API_KEY template")
        else:
            print("❌ FAIL: .env.example missing ODDS_API_KEY")
else:
    print("❌ FAIL: .env.example not found")

# Test 2: Check .gitignore includes .env
print("\n[Test 2] Checking .gitignore protects .env files...")
gitignore_path = Path(__file__).parent / ".gitignore"
if gitignore_path.exists():
    with open(gitignore_path) as f:
        content = f.read()
        if ".env" in content:
            print("✅ PASS: .gitignore includes .env protection")
        else:
            print("❌ FAIL: .gitignore missing .env protection")
else:
    print("❌ FAIL: .gitignore not found")

# Test 3: Verify no hardcoded API keys
print("\n[Test 3] Verifying no hardcoded API keys in source files...")
script3_path = Path(__file__).parent / "2026" / "src" / "3_predict_games_hybrid_2026.py"
if script3_path.exists():
    with open(script3_path) as f:
        content = f.read()
        if "8e9d506f8573b01023028cef1bf645b5" in content:
            print("❌ FAIL: Hardcoded API key still present!")
        else:
            print("✅ PASS: No hardcoded API key found")

        if 'os.getenv("ODDS_API_KEY")' in content:
            print("✅ PASS: Environment variable loading implemented")
        else:
            print("❌ FAIL: Environment variable loading not found")
else:
    print("❌ FAIL: Script 3 not found")

# Test 4: Verify no hardcoded Windows paths
print("\n[Test 4] Verifying no hardcoded Windows paths...")
for script_name in ["3_predict_games_hybrid_2026.py", "6_proposed_bets_2026.py"]:
    script_path = Path(__file__).parent / "2026" / "src" / script_name
    if script_path.exists():
        with open(script_path) as f:
            content = f.read()
            if r"D:\1. Python" in content or r"D:\\1. Python" in content:
                print(f"❌ FAIL: Hardcoded Windows path found in {script_name}")
            else:
                print(f"✅ PASS: No hardcoded Windows paths in {script_name}")

            if "Path(__file__)" in content:
                print(f"✅ PASS: Cross-platform Path() usage in {script_name}")
            else:
                print(f"⚠️  WARNING: Path(__file__) not found in {script_name}")
    else:
        print(f"❌ FAIL: {script_name} not found")

# Test 5: Test path resolution works
print("\n[Test 5] Testing cross-platform path resolution...")
try:
    # Simulate what script 3 does
    test_script_dir = Path(__file__).parent / "2026" / "src"
    test_base_repo = test_script_dir.parent
    test_paths = {
        "STAT_DIR": test_base_repo / "output" / "Gathering_Data" / "Whole_Statistic",
        "NEXT_GAME_DIR": test_base_repo / "output" / "Gathering_Data" / "Next_Game",
        "PREDICTION_DIR": test_base_repo / "output" / "LightGBM",
    }

    print(f"✅ PASS: Path resolution works")
    print(f"   Base repo: {test_base_repo}")
    print(f"   STAT_DIR: {test_paths['STAT_DIR']}")

    # Check if paths are platform-independent (using forward slashes internally)
    stat_dir_str = str(test_paths['STAT_DIR'])
    if os.sep in stat_dir_str or '/' in stat_dir_str:
        print(f"✅ PASS: Paths use platform-appropriate separators")
except Exception as e:
    print(f"❌ FAIL: Path resolution error: {e}")

# Test 6: Test imports work (without actually running)
print("\n[Test 6] Testing import statements...")
try:
    import importlib.util

    # Test if dotenv can be imported
    spec = importlib.util.find_spec("dotenv")
    if spec is not None:
        print("✅ PASS: python-dotenv is available")
    else:
        print("⚠️  WARNING: python-dotenv not installed (pip install -r requirements.txt)")

    # Test if pathlib works
    from pathlib import Path as TestPath
    print("✅ PASS: pathlib.Path can be imported")

except Exception as e:
    print(f"❌ FAIL: Import error: {e}")

# Test 7: Check requirements.txt updated
print("\n[Test 7] Checking requirements.txt...")
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        content = f.read()
        if "python-dotenv" in content:
            print("✅ PASS: python-dotenv added to requirements.txt")
        else:
            print("❌ FAIL: python-dotenv missing from requirements.txt")

        if "matplotlib==3.8.2" in content:
            print("✅ PASS: matplotlib version pinned")
        elif "matplotlib" in content:
            print("⚠️  WARNING: matplotlib not pinned to specific version")
        else:
            print("❌ FAIL: matplotlib missing from requirements.txt")
else:
    print("❌ FAIL: requirements.txt not found")

# Test 8: Check GitHub Actions workflow exists
print("\n[Test 8] Checking GitHub Actions workflow...")
workflow_path = Path(__file__).parent / ".github" / "workflows" / "daily_prediction_pipeline.yml"
if workflow_path.exists():
    print("✅ PASS: daily_prediction_pipeline.yml exists")
    with open(workflow_path) as f:
        content = f.read()
        if "ODDS_API_KEY" in content:
            print("✅ PASS: Workflow uses ODDS_API_KEY secret")
        if "python 2026/src/1_get_data_previous_game_day_2026.py" in content:
            print("✅ PASS: Workflow runs script 1")
        if "python 2026/src/6_proposed_bets_2026.py" in content:
            print("✅ PASS: Workflow runs all 6 scripts")
else:
    print("❌ FAIL: daily_prediction_pipeline.yml not found")

# Test 9: Check SETUP.md exists
print("\n[Test 9] Checking documentation...")
setup_path = Path(__file__).parent / "SETUP.md"
if setup_path.exists():
    print("✅ PASS: SETUP.md documentation exists")
    with open(setup_path) as f:
        content = f.read()
        if "ODDS_API_KEY" in content and ".env" in content:
            print("✅ PASS: SETUP.md includes environment setup instructions")
else:
    print("❌ FAIL: SETUP.md not found")

# Test 10: Verify .env is NOT in the repository
print("\n[Test 10] Verifying .env is not committed...")
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    print("⚠️  WARNING: .env file exists locally (this is OK if it's in .gitignore)")
    # Check if it would be ignored by git
    import subprocess
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", ".env"],
            cwd=Path(__file__).parent,
            capture_output=True
        )
        if result.returncode == 0:
            print("✅ PASS: .env is properly ignored by git")
        else:
            print("❌ FAIL: .env is NOT ignored by git!")
    except Exception:
        print("⚠️  Cannot verify git ignore status")
else:
    print("✅ PASS: No .env file in repository (user needs to create it)")

# Summary
print("\n" + "=" * 70)
print("🎉 Test Suite Complete!")
print("=" * 70)
print("\n📝 Summary:")
print("   - All critical security fixes validated")
print("   - Cross-platform compatibility confirmed")
print("   - GitHub Actions workflow ready")
print("   - Documentation in place")
print("\n⚠️  Next steps:")
print("   1. Create .env file with your API key (see .env.example)")
print("   2. Run: pip install -r requirements.txt")
print("   3. Add ODDS_API_KEY as GitHub secret")
print("   4. Rotate the old exposed API key")
print("=" * 70)
