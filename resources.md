# Resources

## Project-specific
- Mama-mia dataset ([paper](https://www.nature.com/articles/s41597-025-04707-4), [code](https://github.com/LidiaGarrucho/MAMA-MIA/tree/main))
- Segmentation model (small dataset but breast specific: [paper](https://www.nature.com/articles/s41598-024-54048-2), [code](https://github.com/mazurowski-lab/3D-Breast-FGT-and-Blood-Vessel-Segmentation))
  - Jim's fork: https://github.com/jpivarski/3D-Breast-FGT-and-Blood-Vessel-Segmentation
  - The "official" copy in dsi-clinic: https://github.com/dsi-clinic/vanguard-blood-vessel-segmentation
  - Make a fork of this (you will end up with a fork of a fork) and uncheck "Copy the `main` branch only". You will want all branches.
  - Clone your forked repo
  - Switch to the `jpivarski/fix-issues` git branch
  - This is the branch that has a `prediction-workflow.ipynb` notebook file in it (the `main` branch doesn't)
  - Your week 0 task is to get this running and produce the rotating image with blood vessels labeled
- The MAMA-MIA dataset is in `/net/projects2/vanguard` on the cluster
- [Anna's calendly](https://calendly.com/annawoodard) for scheduling 1 on 1s with her
- Schedule a 1 on 1 meeting with Jim in Slack (#2025-autumn-karczmar-lab)
- [project kanban board](https://github.com/orgs/dsi-clinic/projects/6/views/1)
- [project workflow documentation](workflow.md)

## General
- [The Clinic repo](https://github.com/dsi-clinic/the-clinic)  
  General clinic info
- [Slurm tutorial](https://github.com/dsi-clinic/the-clinic/blob/main/tutorials/slurm.md)  
  Instructions for getting set up on the Slurm cluster
- LLMs — use these to understand papers, write docstrings/code, debug, review git diffs, edit writing, and more.  
  In the workplace you’ll compete against others using them, so learning to use them well is essential. ([ChatGPT](https://chat.openai.com), [Claude](https://claude.ai), [Gemini](http://gemini.com), etc.)
- [Grammarly](https://app.grammarly.com/)  
  Helps with grammar and word choice; lighter touch than general-purpose LLMs
- [GitHub flow guide](https://docs.github.com/en/get-started/quickstart/github-flow)  
  The git workflow we are using
- [How to write good commit messages](https://cbea.ms/git-commit/)  

## Domain-specific
- [PyTorch tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html)  
  Basic PyTorch introduction
- [3Blue1Brown neural network series](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=uSlNCw4LFnRVHl7X)  
  Intuitive intro to neural networks
- [PyTorch Geometric colabs](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html)  
  Library for graph neural networks
- [Stanford CS224W (archived)](https://web.archive.org/web/20210620085703/http://web.stanford.edu/class/cs224w/)  
  Class on machine learning with graphs
- [Deep learning YouTube series](https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn)
- [Distill: intro to graph neural networks](https://distill.pub/2021/gnn-intro/)  
