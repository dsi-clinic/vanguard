# Project Workflow Guide
This guide explains our standard process for tracking tasks, writing progress reports, and using GitHub issues & PRs integrated with a project board. It's meant to help you stay organized and make your work traceable.

---

## Core concepts

- **Issue**: A task, bug, or feature to be done.  
- **Assignee**: The person responsible for the issue (assigned when work begins).  
- **Feature branch**: A branch in which you implement the issue.  
- **Pull Request (PR)**: The interface for code review and merging.  
- **Project Board (Kanban)**: Visual status tracker (columns like To Do, In Progress, In Review, Done).  

In short: we create issues → someone picks them up and assigns to themselves → branch off to solve → open PRs → review → merge → automate board updates & issue closure.

---

## Workflow steps

Use this as your checklist when doing work. It helps you—and us—trace every change.

1. **Create an issue**  
   - Give it a clear title, description, acceptance criteria.  
   - **Leave it unassigned** initially — anyone can pick it up later.  
   - Add it to the Project board in the **To Do** column.  

2. **Pick up an issue (when ready to work)**  
   - **Assign it to yourself** to claim ownership.  
   - Move the issue card from **To Do** to **In Progress** on the Kanban board.  
   - Add it to your planning document — just list the issue number (e.g. `#42`).  
   - Now you're officially working on it!

3. **Start a feature branch**  
   - Update your base branch (e.g., `main`).  
   - Create a branch like `feat/42-add-foo` (prefix the issue number).  
   - In commits you can reference the issue (e.g. `Fixes #42 — add foo support`).

4. **Open a Pull Request (PR)**  
   - Open a PR as soon as you have code to show, even if it's not finished (use draft/WIP PR, e.g. "WIP: …").  
   - In the PR description, **link to the issue** (e.g. "Implements #42").  
   - **When the PR is ready for review**, mark it as non-WIP and assign a reviewer.  
   - If you open a PR that's already ready (not WIP), assign a reviewer immediately.

5. **Move to "In Review"**  
   - When the PR is ready for review (marked as non-WIP with a reviewer assigned), move the card in the board from "In Progress" to **In Review**.

6. **Review → Merge**  
   - Reviewers comment, request changes, approve.  
   - If changes are requested, push updates to the same branch (the PR will update automatically).  
   - Once approved and CI/tests pass, merge into the base branch.

7. **Final Automation: Done & Close**  
   - After merging, the card should automatically move to **Done**.  
   - If your PR description used "Fixes #<issue>" or "Closes #<issue>", GitHub will auto-close the issue.  
   - If automation or closing doesn't happen, manually move the card and close the issue.