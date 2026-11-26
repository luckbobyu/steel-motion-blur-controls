# Publishing this artifact

The owner created a private repository and uploaded the project files at
https://github.com/luckbobyu/steel-motion-blur-controls . Root-level file layout
is visible in the supplied owner screenshots. GitHub Actions configuration,
cleanup of a duplicate upload folder and a successful remote check still need
confirmation. Public release also needs author-approved license/redistribution
decisions. Do not send passwords or access tokens in chat.

Repository name: `steel-motion-blur-controls`.

Before publication, check `ARTIFACT_STATUS.json`, `LOCAL_VALIDATION.json` and
`docs/MANUSCRIPT_MAPPING.md`. Confirm that the public description promises only
the capabilities actually verified. The full draft, mock peer reviews, raw lab
feedback bundle, checkpoint binaries and unrelated experiments do not belong in
the initial metric-replay release.

GitHub's repository creation page supports selection of owner, name and
visibility: https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository

The prepared workflow performs count replay/tests on Windows and Linux and an
optional Linux kernel audit. A successful run on GitHub is **not yet confirmed**. Its local
counterparts have been checked, but a green remote run must not be claimed until
the actual Actions jobs complete. Workflow guidance:
https://docs.github.com/en/actions/tutorials/build-and-test-code/python

After the authorized upload, fix any remote discrepancies, record an actual
release/tag and commit, and put that real stable URL in the manuscript. A public
metric-replay release and full independent GPU reexecution are distinct stages.
