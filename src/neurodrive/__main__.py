from neurodrive.cli import main

# The guard matters on Windows: worker processes (SubprocVecEnv, NEAT's parallel
# evaluator) start by re-importing the main module, and must not re-run the CLI.
if __name__ == "__main__":
    raise SystemExit(main())
