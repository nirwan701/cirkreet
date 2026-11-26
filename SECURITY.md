# Security policy

## Reporting a vulnerability

Please do not publish security-sensitive details in a public issue. Use GitHub's **Report a vulnerability** / private vulnerability reporting feature for this repository when it is enabled. Repository maintainers should enable private vulnerability reporting in the repository security settings before making the repository public.

If private reporting is not available, contact the repository maintainers through a private channel listed on the repository profile and provide a concise description, affected components, impact, and reproduction steps. Do not include credentials, tokens, or private circuit/customer data in a report.

## Scope

Security reports may include unsafe file handling, path traversal in import/include handling, code execution, dependency vulnerabilities, or exposure of user-supplied circuit data. The project does not execute arbitrary SPICE behavioral expressions by design; report any behavior that appears to violate that boundary.
