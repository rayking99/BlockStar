"""Optional, import-safe state-conditioned policy training entry point.

Usage: python AgentFormer.py train --dataset results.txt
       python AgentFormer.py sample --checkpoint runs/agentformer.pt
"""

from blockstar.learning import main

if __name__ == "__main__":
    raise SystemExit(main())
