## Commit Message Guidelines

## Commit Message Format

### Structure

* Follow the Conventional Commits


[optional body]

[optional footer(s)]

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to CI configuration files and scripts
- **chore**: Other changes that don't modify src or test files
- **revert**: Reverts a previous commit

### Examples

feat(auth): add OAuth2 authentication
fix(api): resolve null pointer exception in user service
docs: update installation instructions
refactor(utils): simplify date formatting function
test(auth): add unit tests for login validation

### Subject Line (First Line)

- **Length**: Keep under 50 characters
- **Capitalization**: Use lowercase for type and description
- **Tense**: Use imperative mood ("add" not "added" or "adds")
- **Punctuation**: No period at the end
- **Clarity**: Be specific and descriptive

### Body (Optional)

- **Length**: Wrap at 72 characters
- **Content**: Explain what and why, not how
- **Separation**: Leave a blank line between subject and body

### Footer (Optional)

- **Breaking Changes**: Start with "BREAKING CHANGE:"
- **Issue References**: "Closes #123" or "Fixes #456"
- **Co-authors**: "Co-authored-by: Name <email@example.com>"


