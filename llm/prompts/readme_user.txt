Summarize the following code changes from a single commit into a README document.

## Commit Info
- Commit: {commit_hash}
- Repository: {repo_name}

## Changed Seeds (direct code changes)
{seeds_summary}

## Impacted Target Functions
{targets_summary}

## Per-Target Docstring Updates and Analyses
{per_target_details}

## Git Diff
```diff
{diff_text}
```

Generate a complete README document in Markdown format. The README should:
1. Start with a title including the commit hash
2. Provide an executive summary of what this commit does across ALL affected modules
3. List each affected function with its change description and impact
4. End with an overall impact analysis describing how all the changes work together