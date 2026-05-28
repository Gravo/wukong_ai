"""
Create GitHub Issues for wukong_ai project.
Run this script to create 3 key issues.
"""
import json
import urllib.request
import urllib.parse

# Extract token from git remote URL
import subprocess
result = subprocess.run(['git', '-C', 'D:\\projects\\wukong_ai', 'remote', 'get-url', 'origin'], 
                       capture_output=True, text=True)
remote_url = result.stdout.strip()

# Parse token from URL: https://Gravo:ghp_XXX@github.com/...
token = remote_url.split('://')[1].split('@')[0].split(':')[1]
print(f"Token extracted (first 10 chars): {token[:10]}...")

# GitHub API headers
headers = {
    'Authorization': f'token {token}',
    'Accept': 'application/vnd.github.v3+json'
}

# Issue 1: Mouse Input Problem
issue1 = {
    "title": "[Research] Mouse Input Control - Raw Input Blocking",
    "body": """## Problem
Game uses **Raw Input** API to read mouse, bypassing Windows cursor.
- `SendInput` (mouse_util.py) moves Windows cursor, but game doesn't respond
- `pydirectinput` also doesn't work
- No option to disable Raw Input in game settings

## Root Cause
Modern games (2015+) use Raw Input for better precision and anti-cheat.
Black Myth: Wukong is one of them.

## Possible Solutions
1. **Hook Raw Input** - Use C++ DLL injection to simulate Raw Input messages
2. **Driver-level simulation** - Use Interception driver (keyboard/mouse driver interception)
3. **Visual solution** - Don't use mouse events, use OpenCV to recognize minimap direction
4. **Game memory reading** - Read game coordinates (needs reverse engineering)

## References
- [Turing-Project/Black-Myth-Wukong-AI](https://github.com/Turing-Project/Black-Myth-Wukong-AI) (392 stars) - They solved it, need to check their code
- Raw Input documentation: https://docs.microsoft.com/en-us/windows/win32/inputdev/raw-input

## Tasks
- [ ] Research Turing project's mouse control implementation
- [ ] Test Raw Input hook with C++ DLL
- [ ] Evaluate visual solution (OpenCV minimap recognition)
- [ ] Document findings

## Labels
`bug` `help wanted` `research`
""",
    "labels": ["bug", "help wanted", "research"]
}

# Issue 2: Data Imbalance
issue2 = {
    "title": "[Data] Severe Class Imbalance - 87.6% Idle+Forward",
    "body": """## Problem
Training data severely imbalanced:
- idle: 33.7%
- forward: 54.0%
- right: 7.5%
- left: 4.8%
- dodge: 0%

Model degenerates to predicting idle/forward only.

## Attempted Solutions
- ✅ Mouse loss weighting (10x)
- ✅ Rare action weighting (5x)
- ✅ Start frame mouse loss weighting (20x)
- ✅ Direction consistency loss (0.5x)
- ⚠️ Still need more turning data

## Proposed Solutions
1. **Filter idle frames** - Use `filter_idle.py` to remove idle frames
2. **DAgger multi-round** - At least 3-5 rounds until intervention rate < 25%
3. **Data augmentation** - Random crop, color jitter, simulate different lighting
4. **Oversample turning frames** - Duplicate turning samples

## Tasks
- [ ] Run `filter_idle.py` to create balanced dataset
- [ ] Collect more turning data (at least 20% turning samples)
- [ ] DAgger round 2-5 (currently only 1 round)
- [ ] Evaluate effect of data balancing

## Labels
`data` `bug` `help wanted`
""",
    "labels": ["data", "bug", "help wanted"]
}

# Issue 3: Covariate Shift (BC Fundamental Limitation)
issue3 = {
    "title": "[Research] Covariate Shift - BC Fundamental Limitation",
    "body": """## Problem
Behavior Cloning suffers from **covariate shift**:
- Training: expert demonstrations cover limited states
- Inference: model errors accumulate, encounters unseen states
- Result: prediction fails in real game

## Theoretical Analysis
See `docs/RESEARCH_BC_FAILURE_ANALYSIS.md` for details.
Key insight: BC assumes `p_data(state) = p_model(state)`, which doesn't hold.

## Proposed Solutions
1. **DAgger** (Dataset Aggregation) - Already tried, needs more rounds
2. **Goal-Conditioned BC + LSTM** - Temporal modeling, cover historical states
3. **Auxiliary tasks** - Add reconstruction loss, optical flow prediction
4. **Offline RL** - Use Decision Transformer / IQL

## References
- [DAgger paper](https://arxiv.org/abs/1011.0686)
- [Goal-Conditioned BC](https://arxiv.org/abs/1909.11361)
- [Decision Transformer](https://arxiv.org/abs/2106.01345)

## Tasks
- [ ] DAgger round 2-5 (currently only 1 round, intervention rate 71.5%)
- [ ] Implement LSTM version of Goal-Conditioned BC
- [ ] Add auxiliary tasks (reconstruction, optical flow)
- [ ] Evaluate offline RL (Decision Transformer)

## Labels
`research` `enhancement` `help wanted`
""",
    "labels": ["research", "enhancement", "help wanted"]
}

# Create issues
issues = [issue1, issue2, issue3]
created_urls = []

for i, issue in enumerate(issues, 1):
    print(f"\nCreating Issue {i}: {issue['title']}...")
    
    req = urllib.request.Request(
        'https://api.github.com/repos/Gravo/wukong_ai/issues',
        data=json.dumps(issue).encode('utf-8'),
        headers=headers,
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            created_urls.append(result['html_url'])
            print(f"  ✅ Created: {result['html_url']}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

print("\n" + "=" * 60)
print("Summary:")
for url in created_urls:
    print(f"  - {url}")
print("=" * 60)
