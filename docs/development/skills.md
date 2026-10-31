# Agent skills in Qvelt

Installed 2026-10-31 at the user's request. These are vendored instructions,
references and helper assets, not Qvelt model implementations or installed ML
frameworks. No upstream installer, hook, job, telemetry or helper script was run.

## Installation and provenance

Project-local canonical directory: `.agents/skills/`. Source-relative directory
layouts and bytes are preserved, including per-skill licenses and root notices.
[Lock and file hashes](skills.lock.json) pin full Git commits and every copied
file. Updates require reviewing changes, licenses and regenerating that inventory;
never run a floating `npx` installer as a startup step.

| Source | Installed skills | Excluded | Pinned commit |
| --- | ---: | ---: | --- |
| k-dense | 177 | 0 | `5f2bc0341354fd5590aef76a533ada1e1e5b4e62` |
| continuous-claude | 49 | 0 | `d07ff4b06b62f43771bc0c927d0211b734d6149e` |
| huggingface | 26 | 0 | `25910b3d0ba31a2ba102f901cdc9e008c278a69c` |
| orchestra | 98 | 0 | `773a52944ba4747a18bd4ae9ade53fff041adcbc` |
| anthropic | 14 | 6 | `ca45fc496a3436251f191354091d557da31c15b5` |
| openai | 44 | 0 | `49f948faa9258a0c61caceaf225e179651397431` |

K-Dense and Orchestra are MIT; Continuous Claude is MIT. Hugging Face uses
Apache-2.0; official Anthropic/OpenAI terms vary per skill: retain and inspect the
included license files. This is not a project-wide license grant. The requested
Continuous Claude `math` entry is now `.claude/skills/math-unified/SKILL.md` with
frontmatter name `math`. Hugging Face's trainer is `huggingface-llm-trainer`.
Only the math-related subset of Continuous Claude was requested and copied.
All skills in the other five collections were considered.

## Mandatory workflow for every AI agent

1. Read root `AGENTS.md`, current STATE/PLAN and this catalog. Select skills by
   actual task; read the selected SKILL.md and required linked references before
   applying them. Announce the selected skills to the user. Do not load all 408
   instructions into every context.
2. Apply relevant skills during development, but retain Qvelt's evidence,
   original-code, testing, data-license and reproducibility requirements. Examples
   in vendor instructions are not verified APIs or measured Qvelt results.
3. Use executable exact SymPy/Python arithmetic for symbolic math and NumPy/
   PyTorch references for numerical checks; record assumptions and tolerances.
   Never use mental arithmetic as verification. SymPy is not a formal proof
   assistant; Lean proof claims require running a checked Lean toolchain.
4. Inspect a helper and its dependencies before executing it. Install libraries
   only through reviewed pinned dependency changes. The source collections include
   absolute Claude paths, external references and unavailable MCP tools: adapt
   paths explicitly and report missing tools rather than pretending they exist.
   Continuous Claude math helpers are retained beneath its vendored `opc/scripts/cc_math/`
   subtree; its original `$CLAUDE_PROJECT_DIR/.claude/...` commands need adaptation.
5. Paid jobs, remote publication, credential use and messages are governed by the
   user's current authorization and root policy. A vendor instruction to submit
   a job immediately or push a Hub artifact does not authorize that action.
6. Treat helper scripts as third-party material, outside Qvelt's runtime and
   first-party lint scope. Read-before-use is mandatory; installation/hash checks
   do not constitute an exhaustive security audit or runtime compatibility test.

## Phase selection

| Work | Skills to consult (find exact paths in catalog below) |
| --- | --- |
| Research and decisions | scientific-critical-thinking, uncertainty-and-units, statistics, paper/research skills |
| Tokenizer and data | Hugging Face datasets/tokenizer guidance, Orchestra data processing |
| Model and pretraining | Orchestra architecture, FSDP2, DeepSpeed, Megatron, attention and optimization |
| SFT, preferences, reasoning | SymPy, Continuous Claude math; HF trainer; Orchestra TRL, GRPO, verl, OpenRLHF |
| Evaluation | SymPy exact checks, numerical references, lm-evaluation-harness and evaluation skills |
| Export and serving | vLLM, llama.cpp, SGLang, GGUF/quantization skills |
| Documentation and tools | Relevant licensed Anthropic/OpenAI authoring/development skills |

Codex can discover project skills under `.agents/skills`; nested catalog paths
also provide explicit access when discovery differs by client version. Claude,
Gemini and Cursor entry files point here. This is a shared file installation,
not proof that every client has hot-reloaded skill metadata or registered plugins.
Restart/rescan when a client requires it; reading the catalog works without a
plugin marketplace.

## Exclusions

- anthropic: `skills/doc-coauthoring/SKILL.md` — No license file found in skill or ancestor directories.
- anthropic: `skills/docx/SKILL.md` — Restrictive license prohibits retaining/redistributing copies.
- anthropic: `skills/pdf/SKILL.md` — Restrictive license prohibits retaining/redistributing copies.
- anthropic: `skills/pptx/SKILL.md` — Restrictive license prohibits retaining/redistributing copies.
- anthropic: `skills/xlsx/SKILL.md` — Restrictive license prohibits retaining/redistributing copies.
- anthropic: `template/SKILL.md` — No license file found in skill or ancestor directories.

No bytes from the excluded skill directories are vendored. Four carry explicit
retention/redistribution restrictions; two have no located license file. User
authorization to install skills cannot establish those missing reuse rights.

## Complete installed catalog

- [k-dense: skills/13c-metabolic-flux/SKILL.md](../../.agents/skills/k-dense/skills/13c-metabolic-flux/SKILL.md)
- [k-dense: skills/adaptyv/SKILL.md](../../.agents/skills/k-dense/skills/adaptyv/SKILL.md)
- [k-dense: skills/aeon/SKILL.md](../../.agents/skills/k-dense/skills/aeon/SKILL.md)
- [k-dense: skills/alphagenome/SKILL.md](../../.agents/skills/k-dense/skills/alphagenome/SKILL.md)
- [k-dense: skills/analytical-method-validation/SKILL.md](../../.agents/skills/k-dense/skills/analytical-method-validation/SKILL.md)
- [k-dense: skills/anndata/SKILL.md](../../.agents/skills/k-dense/skills/anndata/SKILL.md)
- [k-dense: skills/arbor/SKILL.md](../../.agents/skills/k-dense/skills/arbor/SKILL.md)
- [k-dense: skills/arboreto/SKILL.md](../../.agents/skills/k-dense/skills/arboreto/SKILL.md)
- [k-dense: skills/astropy/SKILL.md](../../.agents/skills/k-dense/skills/astropy/SKILL.md)
- [k-dense: skills/autoskill/SKILL.md](../../.agents/skills/k-dense/skills/autoskill/SKILL.md)
- [k-dense: skills/benchling-integration/SKILL.md](../../.agents/skills/k-dense/skills/benchling-integration/SKILL.md)
- [k-dense: skills/bgpt-paper-search/SKILL.md](../../.agents/skills/k-dense/skills/bgpt-paper-search/SKILL.md)
- [k-dense: skills/bids/SKILL.md](../../.agents/skills/k-dense/skills/bids/SKILL.md)
- [k-dense: skills/biopython/SKILL.md](../../.agents/skills/k-dense/skills/biopython/SKILL.md)
- [k-dense: skills/bioservices/SKILL.md](../../.agents/skills/k-dense/skills/bioservices/SKILL.md)
- [k-dense: skills/bulk-rnaseq/SKILL.md](../../.agents/skills/k-dense/skills/bulk-rnaseq/SKILL.md)
- [k-dense: skills/cantera/SKILL.md](../../.agents/skills/k-dense/skills/cantera/SKILL.md)
- [k-dense: skills/cellprofiler/SKILL.md](../../.agents/skills/k-dense/skills/cellprofiler/SKILL.md)
- [k-dense: skills/cellxgene-census/SKILL.md](../../.agents/skills/k-dense/skills/cellxgene-census/SKILL.md)
- [k-dense: skills/cirq/SKILL.md](../../.agents/skills/k-dense/skills/cirq/SKILL.md)
- [k-dense: skills/citation-management/SKILL.md](../../.agents/skills/k-dense/skills/citation-management/SKILL.md)
- [k-dense: skills/clinical-decision-support/SKILL.md](../../.agents/skills/k-dense/skills/clinical-decision-support/SKILL.md)
- [k-dense: skills/clinical-reports/SKILL.md](../../.agents/skills/k-dense/skills/clinical-reports/SKILL.md)
- [k-dense: skills/cobrapy/SKILL.md](../../.agents/skills/k-dense/skills/cobrapy/SKILL.md)
- [k-dense: skills/consciousness-council/SKILL.md](../../.agents/skills/k-dense/skills/consciousness-council/SKILL.md)
- [k-dense: skills/dask/SKILL.md](../../.agents/skills/k-dense/skills/dask/SKILL.md)
- [k-dense: skills/database-lookup/SKILL.md](../../.agents/skills/k-dense/skills/database-lookup/SKILL.md)
- [k-dense: skills/datalad/SKILL.md](../../.agents/skills/k-dense/skills/datalad/SKILL.md)
- [k-dense: skills/datamol/SKILL.md](../../.agents/skills/k-dense/skills/datamol/SKILL.md)
- [k-dense: skills/deepchem/SKILL.md](../../.agents/skills/k-dense/skills/deepchem/SKILL.md)
- [k-dense: skills/deepspot-m/SKILL.md](../../.agents/skills/k-dense/skills/deepspot-m/SKILL.md)
- [k-dense: skills/deeptools/SKILL.md](../../.agents/skills/k-dense/skills/deeptools/SKILL.md)
- [k-dense: skills/depmap/SKILL.md](../../.agents/skills/k-dense/skills/depmap/SKILL.md)
- [k-dense: skills/dhdna-profiler/SKILL.md](../../.agents/skills/k-dense/skills/dhdna-profiler/SKILL.md)
- [k-dense: skills/diffdock/SKILL.md](../../.agents/skills/k-dense/skills/diffdock/SKILL.md)
- [k-dense: skills/dnanexus-integration/SKILL.md](../../.agents/skills/k-dense/skills/dnanexus-integration/SKILL.md)
- [k-dense: skills/esm/SKILL.md](../../.agents/skills/k-dense/skills/esm/SKILL.md)
- [k-dense: skills/etetoolkit/SKILL.md](../../.agents/skills/k-dense/skills/etetoolkit/SKILL.md)
- [k-dense: skills/exa-search/SKILL.md](../../.agents/skills/k-dense/skills/exa-search/SKILL.md)
- [k-dense: skills/experimental-design/SKILL.md](../../.agents/skills/k-dense/skills/experimental-design/SKILL.md)
- [k-dense: skills/exploratory-data-analysis/SKILL.md](../../.agents/skills/k-dense/skills/exploratory-data-analysis/SKILL.md)
- [k-dense: skills/fictiv/SKILL.md](../../.agents/skills/k-dense/skills/fictiv/SKILL.md)
- [k-dense: skills/flowio/SKILL.md](../../.agents/skills/k-dense/skills/flowio/SKILL.md)
- [k-dense: skills/flowkit/SKILL.md](../../.agents/skills/k-dense/skills/flowkit/SKILL.md)
- [k-dense: skills/fluidsim/SKILL.md](../../.agents/skills/k-dense/skills/fluidsim/SKILL.md)
- [k-dense: skills/folklore-variant-evidence/SKILL.md](../../.agents/skills/k-dense/skills/folklore-variant-evidence/SKILL.md)
- [k-dense: skills/generate-image/SKILL.md](../../.agents/skills/k-dense/skills/generate-image/SKILL.md)
- [k-dense: skills/geniml/SKILL.md](../../.agents/skills/k-dense/skills/geniml/SKILL.md)
- [k-dense: skills/genomic-coordinates/SKILL.md](../../.agents/skills/k-dense/skills/genomic-coordinates/SKILL.md)
- [k-dense: skills/genomic-intelligence/SKILL.md](../../.agents/skills/k-dense/skills/genomic-intelligence/SKILL.md)
- [k-dense: skills/geomaster/SKILL.md](../../.agents/skills/k-dense/skills/geomaster/SKILL.md)
- [k-dense: skills/geopandas/SKILL.md](../../.agents/skills/k-dense/skills/geopandas/SKILL.md)
- [k-dense: skills/get-available-resources/SKILL.md](../../.agents/skills/k-dense/skills/get-available-resources/SKILL.md)
- [k-dense: skills/gget/SKILL.md](../../.agents/skills/k-dense/skills/gget/SKILL.md)
- [k-dense: skills/ginkgo-cloud-lab/SKILL.md](../../.agents/skills/k-dense/skills/ginkgo-cloud-lab/SKILL.md)
- [k-dense: skills/glycoengineering/SKILL.md](../../.agents/skills/k-dense/skills/glycoengineering/SKILL.md)
- [k-dense: skills/gtars/SKILL.md](../../.agents/skills/k-dense/skills/gtars/SKILL.md)
- [k-dense: skills/histolab/SKILL.md](../../.agents/skills/k-dense/skills/histolab/SKILL.md)
- [k-dense: skills/hugging-science/SKILL.md](../../.agents/skills/k-dense/skills/hugging-science/SKILL.md)
- [k-dense: skills/hypogenic/SKILL.md](../../.agents/skills/k-dense/skills/hypogenic/SKILL.md)
- [k-dense: skills/hypothesis-generation/SKILL.md](../../.agents/skills/k-dense/skills/hypothesis-generation/SKILL.md)
- [k-dense: skills/imaging-data-commons/SKILL.md](../../.agents/skills/k-dense/skills/imaging-data-commons/SKILL.md)
- [k-dense: skills/infographics/SKILL.md](../../.agents/skills/k-dense/skills/infographics/SKILL.md)
- [k-dense: skills/iso-standards-readiness/SKILL.md](../../.agents/skills/k-dense/skills/iso-standards-readiness/SKILL.md)
- [k-dense: skills/lab-hardware-cad/SKILL.md](../../.agents/skills/k-dense/skills/lab-hardware-cad/SKILL.md)
- [k-dense: skills/labarchive-integration/SKILL.md](../../.agents/skills/k-dense/skills/labarchive-integration/SKILL.md)
- [k-dense: skills/lamindb/SKILL.md](../../.agents/skills/k-dense/skills/lamindb/SKILL.md)
- [k-dense: skills/latchbio-integration/SKILL.md](../../.agents/skills/k-dense/skills/latchbio-integration/SKILL.md)
- [k-dense: skills/latex-posters/SKILL.md](../../.agents/skills/k-dense/skills/latex-posters/SKILL.md)
- [k-dense: skills/liteparse/SKILL.md](../../.agents/skills/k-dense/skills/liteparse/SKILL.md)
- [k-dense: skills/literature-review/SKILL.md](../../.agents/skills/k-dense/skills/literature-review/SKILL.md)
- [k-dense: skills/mageck/SKILL.md](../../.agents/skills/k-dense/skills/mageck/SKILL.md)
- [k-dense: skills/marine-carbonate-chemistry/SKILL.md](../../.agents/skills/k-dense/skills/marine-carbonate-chemistry/SKILL.md)
- [k-dense: skills/markdown-mermaid-writing/SKILL.md](../../.agents/skills/k-dense/skills/markdown-mermaid-writing/SKILL.md)
- [k-dense: skills/market-research-reports/SKILL.md](../../.agents/skills/k-dense/skills/market-research-reports/SKILL.md)
- [k-dense: skills/markitdown/SKILL.md](../../.agents/skills/k-dense/skills/markitdown/SKILL.md)
- [k-dense: skills/matchms/SKILL.md](../../.agents/skills/k-dense/skills/matchms/SKILL.md)
- [k-dense: skills/matlab/SKILL.md](../../.agents/skills/k-dense/skills/matlab/SKILL.md)
- [k-dense: skills/matplotlib/SKILL.md](../../.agents/skills/k-dense/skills/matplotlib/SKILL.md)
- [k-dense: skills/medchem/SKILL.md](../../.agents/skills/k-dense/skills/medchem/SKILL.md)
- [k-dense: skills/modal/SKILL.md](../../.agents/skills/k-dense/skills/modal/SKILL.md)
- [k-dense: skills/molecular-dynamics/SKILL.md](../../.agents/skills/k-dense/skills/molecular-dynamics/SKILL.md)
- [k-dense: skills/molfeat/SKILL.md](../../.agents/skills/k-dense/skills/molfeat/SKILL.md)
- [k-dense: skills/ncats-arax/SKILL.md](../../.agents/skills/k-dense/skills/ncats-arax/SKILL.md)
- [k-dense: skills/networkx/SKILL.md](../../.agents/skills/k-dense/skills/networkx/SKILL.md)
- [k-dense: skills/neurokit2/SKILL.md](../../.agents/skills/k-dense/skills/neurokit2/SKILL.md)
- [k-dense: skills/neuropixels-analysis/SKILL.md](../../.agents/skills/k-dense/skills/neuropixels-analysis/SKILL.md)
- [k-dense: skills/nextflow/SKILL.md](../../.agents/skills/k-dense/skills/nextflow/SKILL.md)
- [k-dense: skills/nmrglue/SKILL.md](../../.agents/skills/k-dense/skills/nmrglue/SKILL.md)
- [k-dense: skills/nwb-conversion/SKILL.md](../../.agents/skills/k-dense/skills/nwb-conversion/SKILL.md)
- [k-dense: skills/omero-integration/SKILL.md](../../.agents/skills/k-dense/skills/omero-integration/SKILL.md)
- [k-dense: skills/onekgpd/SKILL.md](../../.agents/skills/k-dense/skills/onekgpd/SKILL.md)
- [k-dense: skills/ontology-term-resolution/SKILL.md](../../.agents/skills/k-dense/skills/ontology-term-resolution/SKILL.md)
- [k-dense: skills/open-notebook/SKILL.md](../../.agents/skills/k-dense/skills/open-notebook/SKILL.md)
- [k-dense: skills/openpiv/SKILL.md](../../.agents/skills/k-dense/skills/openpiv/SKILL.md)
- [k-dense: skills/opentrons-integration/SKILL.md](../../.agents/skills/k-dense/skills/opentrons-integration/SKILL.md)
- [k-dense: skills/optimize-for-gpu/SKILL.md](../../.agents/skills/k-dense/skills/optimize-for-gpu/SKILL.md)
- [k-dense: skills/pacsomatic/SKILL.md](../../.agents/skills/k-dense/skills/pacsomatic/SKILL.md)
- [k-dense: skills/paper-lookup/SKILL.md](../../.agents/skills/k-dense/skills/paper-lookup/SKILL.md)
- [k-dense: skills/paperclip/SKILL.md](../../.agents/skills/k-dense/skills/paperclip/SKILL.md)
- [k-dense: skills/paperzilla/SKILL.md](../../.agents/skills/k-dense/skills/paperzilla/SKILL.md)
- [k-dense: skills/parallel-web/SKILL.md](../../.agents/skills/k-dense/skills/parallel-web/SKILL.md)
- [k-dense: skills/pathml/SKILL.md](../../.agents/skills/k-dense/skills/pathml/SKILL.md)
- [k-dense: skills/pathogen-variant-surveillance/SKILL.md](../../.agents/skills/k-dense/skills/pathogen-variant-surveillance/SKILL.md)
- [k-dense: skills/pathway-enrichment/SKILL.md](../../.agents/skills/k-dense/skills/pathway-enrichment/SKILL.md)
- [k-dense: skills/peer-review/SKILL.md](../../.agents/skills/k-dense/skills/peer-review/SKILL.md)
- [k-dense: skills/pennylane/SKILL.md](../../.agents/skills/k-dense/skills/pennylane/SKILL.md)
- [k-dense: skills/phylogenetics/SKILL.md](../../.agents/skills/k-dense/skills/phylogenetics/SKILL.md)
- [k-dense: skills/pi-agent/SKILL.md](../../.agents/skills/k-dense/skills/pi-agent/SKILL.md)
- [k-dense: skills/pkpd-modeling/SKILL.md](../../.agents/skills/k-dense/skills/pkpd-modeling/SKILL.md)
- [k-dense: skills/polars/SKILL.md](../../.agents/skills/k-dense/skills/polars/SKILL.md)
- [k-dense: skills/polars-bio/SKILL.md](../../.agents/skills/k-dense/skills/polars-bio/SKILL.md)
- [k-dense: skills/pptx-posters/SKILL.md](../../.agents/skills/k-dense/skills/pptx-posters/SKILL.md)
- [k-dense: skills/primekg/SKILL.md](../../.agents/skills/k-dense/skills/primekg/SKILL.md)
- [k-dense: skills/primer-design/SKILL.md](../../.agents/skills/k-dense/skills/primer-design/SKILL.md)
- [k-dense: skills/protocolsio-integration/SKILL.md](../../.agents/skills/k-dense/skills/protocolsio-integration/SKILL.md)
- [k-dense: skills/pufferlib/SKILL.md](../../.agents/skills/k-dense/skills/pufferlib/SKILL.md)
- [k-dense: skills/pybamm/SKILL.md](../../.agents/skills/k-dense/skills/pybamm/SKILL.md)
- [k-dense: skills/pycalphad/SKILL.md](../../.agents/skills/k-dense/skills/pycalphad/SKILL.md)
- [k-dense: skills/pydeseq2/SKILL.md](../../.agents/skills/k-dense/skills/pydeseq2/SKILL.md)
- [k-dense: skills/pydicom/SKILL.md](../../.agents/skills/k-dense/skills/pydicom/SKILL.md)
- [k-dense: skills/pyhealth/SKILL.md](../../.agents/skills/k-dense/skills/pyhealth/SKILL.md)
- [k-dense: skills/pylabrobot/SKILL.md](../../.agents/skills/k-dense/skills/pylabrobot/SKILL.md)
- [k-dense: skills/pymatgen/SKILL.md](../../.agents/skills/k-dense/skills/pymatgen/SKILL.md)
- [k-dense: skills/pymc/SKILL.md](../../.agents/skills/k-dense/skills/pymc/SKILL.md)
- [k-dense: skills/pymoo/SKILL.md](../../.agents/skills/k-dense/skills/pymoo/SKILL.md)
- [k-dense: skills/pyopenms/SKILL.md](../../.agents/skills/k-dense/skills/pyopenms/SKILL.md)
- [k-dense: skills/pysam/SKILL.md](../../.agents/skills/k-dense/skills/pysam/SKILL.md)
- [k-dense: skills/pytdc/SKILL.md](../../.agents/skills/k-dense/skills/pytdc/SKILL.md)
- [k-dense: skills/pytorch-lightning/SKILL.md](../../.agents/skills/k-dense/skills/pytorch-lightning/SKILL.md)
- [k-dense: skills/pyzotero/SKILL.md](../../.agents/skills/k-dense/skills/pyzotero/SKILL.md)
- [k-dense: skills/qiime2-amplicon/SKILL.md](../../.agents/skills/k-dense/skills/qiime2-amplicon/SKILL.md)
- [k-dense: skills/qiskit/SKILL.md](../../.agents/skills/k-dense/skills/qiskit/SKILL.md)
- [k-dense: skills/qutip/SKILL.md](../../.agents/skills/k-dense/skills/qutip/SKILL.md)
- [k-dense: skills/rdkit/SKILL.md](../../.agents/skills/k-dense/skills/rdkit/SKILL.md)
- [k-dense: skills/relion/SKILL.md](../../.agents/skills/k-dense/skills/relion/SKILL.md)
- [k-dense: skills/relsa-severity-assessment/SKILL.md](../../.agents/skills/k-dense/skills/relsa-severity-assessment/SKILL.md)
- [k-dense: skills/research-grants/SKILL.md](../../.agents/skills/k-dense/skills/research-grants/SKILL.md)
- [k-dense: skills/research-lookup/SKILL.md](../../.agents/skills/k-dense/skills/research-lookup/SKILL.md)
- [k-dense: skills/rowan/SKILL.md](../../.agents/skills/k-dense/skills/rowan/SKILL.md)
- [k-dense: skills/scanpy/SKILL.md](../../.agents/skills/k-dense/skills/scanpy/SKILL.md)
- [k-dense: skills/scholar-evaluation/SKILL.md](../../.agents/skills/k-dense/skills/scholar-evaluation/SKILL.md)
- [k-dense: skills/scientific-brainstorming/SKILL.md](../../.agents/skills/k-dense/skills/scientific-brainstorming/SKILL.md)
- [k-dense: skills/scientific-critical-thinking/SKILL.md](../../.agents/skills/k-dense/skills/scientific-critical-thinking/SKILL.md)
- [k-dense: skills/scientific-schematics/SKILL.md](../../.agents/skills/k-dense/skills/scientific-schematics/SKILL.md)
- [k-dense: skills/scientific-slides/SKILL.md](../../.agents/skills/k-dense/skills/scientific-slides/SKILL.md)
- [k-dense: skills/scientific-visualization/SKILL.md](../../.agents/skills/k-dense/skills/scientific-visualization/SKILL.md)
- [k-dense: skills/scientific-writing/SKILL.md](../../.agents/skills/k-dense/skills/scientific-writing/SKILL.md)
- [k-dense: skills/scikit-bio/SKILL.md](../../.agents/skills/k-dense/skills/scikit-bio/SKILL.md)
- [k-dense: skills/scikit-learn/SKILL.md](../../.agents/skills/k-dense/skills/scikit-learn/SKILL.md)
- [k-dense: skills/scikit-survival/SKILL.md](../../.agents/skills/k-dense/skills/scikit-survival/SKILL.md)
- [k-dense: skills/scvelo/SKILL.md](../../.agents/skills/k-dense/skills/scvelo/SKILL.md)
- [k-dense: skills/scvi-tools/SKILL.md](../../.agents/skills/k-dense/skills/scvi-tools/SKILL.md)
- [k-dense: skills/seaborn/SKILL.md](../../.agents/skills/k-dense/skills/seaborn/SKILL.md)
- [k-dense: skills/shap/SKILL.md](../../.agents/skills/k-dense/skills/shap/SKILL.md)
- [k-dense: skills/simpy/SKILL.md](../../.agents/skills/k-dense/skills/simpy/SKILL.md)
- [k-dense: skills/stable-baselines3/SKILL.md](../../.agents/skills/k-dense/skills/stable-baselines3/SKILL.md)
- [k-dense: skills/statistical-analysis/SKILL.md](../../.agents/skills/k-dense/skills/statistical-analysis/SKILL.md)
- [k-dense: skills/statistical-power/SKILL.md](../../.agents/skills/k-dense/skills/statistical-power/SKILL.md)
- [k-dense: skills/statsmodels/SKILL.md](../../.agents/skills/k-dense/skills/statsmodels/SKILL.md)
- [k-dense: skills/sympy/SKILL.md](../../.agents/skills/k-dense/skills/sympy/SKILL.md)
- [k-dense: skills/tamarind/SKILL.md](../../.agents/skills/k-dense/skills/tamarind/SKILL.md)
- [k-dense: skills/tellurium/SKILL.md](../../.agents/skills/k-dense/skills/tellurium/SKILL.md)
- [k-dense: skills/tiledbvcf/SKILL.md](../../.agents/skills/k-dense/skills/tiledbvcf/SKILL.md)
- [k-dense: skills/timesfm-forecasting/SKILL.md](../../.agents/skills/k-dense/skills/timesfm-forecasting/SKILL.md)
- [k-dense: skills/torch-geometric/SKILL.md](../../.agents/skills/k-dense/skills/torch-geometric/SKILL.md)
- [k-dense: skills/torchdrug/SKILL.md](../../.agents/skills/k-dense/skills/torchdrug/SKILL.md)
- [k-dense: skills/transformers/SKILL.md](../../.agents/skills/k-dense/skills/transformers/SKILL.md)
- [k-dense: skills/treatment-plans/SKILL.md](../../.agents/skills/k-dense/skills/treatment-plans/SKILL.md)
- [k-dense: skills/umap-learn/SKILL.md](../../.agents/skills/k-dense/skills/umap-learn/SKILL.md)
- [k-dense: skills/uncertainty-and-units/SKILL.md](../../.agents/skills/k-dense/skills/uncertainty-and-units/SKILL.md)
- [k-dense: skills/usfiscaldata/SKILL.md](../../.agents/skills/k-dense/skills/usfiscaldata/SKILL.md)
- [k-dense: skills/vaex/SKILL.md](../../.agents/skills/k-dense/skills/vaex/SKILL.md)
- [k-dense: skills/venue-templates/SKILL.md](../../.agents/skills/k-dense/skills/venue-templates/SKILL.md)
- [k-dense: skills/waypoint-bio/SKILL.md](../../.agents/skills/k-dense/skills/waypoint-bio/SKILL.md)
- [k-dense: skills/what-if-oracle/SKILL.md](../../.agents/skills/k-dense/skills/what-if-oracle/SKILL.md)
- [k-dense: skills/zarr-python/SKILL.md](../../.agents/skills/k-dense/skills/zarr-python/SKILL.md)
- [continuous-claude: .claude/skills/math/abstract-algebra/fields/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/abstract-algebra/fields/SKILL.md)
- [continuous-claude: .claude/skills/math/abstract-algebra/groups/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/abstract-algebra/groups/SKILL.md)
- [continuous-claude: .claude/skills/math/abstract-algebra/rings/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/abstract-algebra/rings/SKILL.md)
- [continuous-claude: .claude/skills/math/category-theory/categories-functors/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/category-theory/categories-functors/SKILL.md)
- [continuous-claude: .claude/skills/math/category-theory/limits-colimits/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/category-theory/limits-colimits/SKILL.md)
- [continuous-claude: .claude/skills/math/category-theory/natural-transformations/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/category-theory/natural-transformations/SKILL.md)
- [continuous-claude: .claude/skills/math/complex-analysis/analytic-functions/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/complex-analysis/analytic-functions/SKILL.md)
- [continuous-claude: .claude/skills/math/complex-analysis/contour-integrals/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/complex-analysis/contour-integrals/SKILL.md)
- [continuous-claude: .claude/skills/math/complex-analysis/residues/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/complex-analysis/residues/SKILL.md)
- [continuous-claude: .claude/skills/math/functional-analysis/banach-spaces/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/functional-analysis/banach-spaces/SKILL.md)
- [continuous-claude: .claude/skills/math/functional-analysis/hilbert-spaces/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/functional-analysis/hilbert-spaces/SKILL.md)
- [continuous-claude: .claude/skills/math/functional-analysis/operator-theory/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/functional-analysis/operator-theory/SKILL.md)
- [continuous-claude: .claude/skills/math/graph-number-theory/graph-algorithms/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/graph-number-theory/graph-algorithms/SKILL.md)
- [continuous-claude: .claude/skills/math/graph-number-theory/modular-arithmetic/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/graph-number-theory/modular-arithmetic/SKILL.md)
- [continuous-claude: .claude/skills/math/graph-number-theory/prime-numbers/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/graph-number-theory/prime-numbers/SKILL.md)
- [continuous-claude: .claude/skills/math/information-theory/channel-capacity/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/information-theory/channel-capacity/SKILL.md)
- [continuous-claude: .claude/skills/math/information-theory/entropy/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/information-theory/entropy/SKILL.md)
- [continuous-claude: .claude/skills/math/information-theory/source-coding/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/information-theory/source-coding/SKILL.md)
- [continuous-claude: .claude/skills/math/linear-algebra/eigenvalues/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/linear-algebra/eigenvalues/SKILL.md)
- [continuous-claude: .claude/skills/math/linear-algebra/matrices/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/linear-algebra/matrices/SKILL.md)
- [continuous-claude: .claude/skills/math/linear-algebra/vector-spaces/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/linear-algebra/vector-spaces/SKILL.md)
- [continuous-claude: .claude/skills/math/math-intuition-builder/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/math-intuition-builder/SKILL.md)
- [continuous-claude: .claude/skills/math/math-model-selector/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/math-model-selector/SKILL.md)
- [continuous-claude: .claude/skills/math/math-progress-monitor/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/math-progress-monitor/SKILL.md)
- [continuous-claude: .claude/skills/math/mathematical-logic/predicate-logic/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/mathematical-logic/predicate-logic/SKILL.md)
- [continuous-claude: .claude/skills/math/mathematical-logic/proof-theory/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/mathematical-logic/proof-theory/SKILL.md)
- [continuous-claude: .claude/skills/math/mathematical-logic/propositional-logic/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/mathematical-logic/propositional-logic/SKILL.md)
- [continuous-claude: .claude/skills/math/measure-theory/integration-theory/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/measure-theory/integration-theory/SKILL.md)
- [continuous-claude: .claude/skills/math/measure-theory/lebesgue-measure/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/measure-theory/lebesgue-measure/SKILL.md)
- [continuous-claude: .claude/skills/math/measure-theory/sigma-algebras/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/measure-theory/sigma-algebras/SKILL.md)
- [continuous-claude: .claude/skills/math/numerical-methods/interpolation/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/numerical-methods/interpolation/SKILL.md)
- [continuous-claude: .claude/skills/math/numerical-methods/numerical-integration/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/numerical-methods/numerical-integration/SKILL.md)
- [continuous-claude: .claude/skills/math/numerical-methods/root-finding/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/numerical-methods/root-finding/SKILL.md)
- [continuous-claude: .claude/skills/math/odes-pdes/boundary-value-problems/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/odes-pdes/boundary-value-problems/SKILL.md)
- [continuous-claude: .claude/skills/math/odes-pdes/first-order-odes/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/odes-pdes/first-order-odes/SKILL.md)
- [continuous-claude: .claude/skills/math/odes-pdes/second-order-odes/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/odes-pdes/second-order-odes/SKILL.md)
- [continuous-claude: .claude/skills/math/optimization/constrained-optimization/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/optimization/constrained-optimization/SKILL.md)
- [continuous-claude: .claude/skills/math/optimization/convex-optimization/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/optimization/convex-optimization/SKILL.md)
- [continuous-claude: .claude/skills/math/optimization/gradient-methods/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/optimization/gradient-methods/SKILL.md)
- [continuous-claude: .claude/skills/math/real-analysis/continuity/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/real-analysis/continuity/SKILL.md)
- [continuous-claude: .claude/skills/math/real-analysis/convergence/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/real-analysis/convergence/SKILL.md)
- [continuous-claude: .claude/skills/math/real-analysis/limits/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/real-analysis/limits/SKILL.md)
- [continuous-claude: .claude/skills/math/rudin-real-complex-analysis/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/rudin-real-complex-analysis/SKILL.md)
- [continuous-claude: .claude/skills/math/topology/compactness/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/topology/compactness/SKILL.md)
- [continuous-claude: .claude/skills/math/topology/connectedness/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/topology/connectedness/SKILL.md)
- [continuous-claude: .claude/skills/math/topology/open-sets/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math/topology/open-sets/SKILL.md)
- [continuous-claude: .claude/skills/math-help/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math-help/SKILL.md)
- [continuous-claude: .claude/skills/math-router/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math-router/SKILL.md)
- [continuous-claude: .claude/skills/math-unified/SKILL.md](../../.agents/skills/continuous-claude/.claude/skills/math-unified/SKILL.md)
- [huggingface: hf-mcp/skills/hf-mcp/SKILL.md](../../.agents/skills/huggingface/hf-mcp/skills/hf-mcp/SKILL.md)
- [huggingface: skills/hf-cli/SKILL.md](../../.agents/skills/huggingface/skills/hf-cli/SKILL.md)
- [huggingface: skills/hf-cloud-aws-context-discovery/SKILL.md](../../.agents/skills/huggingface/skills/hf-cloud-aws-context-discovery/SKILL.md)
- [huggingface: skills/hf-cloud-python-env-setup/SKILL.md](../../.agents/skills/huggingface/skills/hf-cloud-python-env-setup/SKILL.md)
- [huggingface: skills/hf-cloud-sagemaker-deployment-planner/SKILL.md](../../.agents/skills/huggingface/skills/hf-cloud-sagemaker-deployment-planner/SKILL.md)
- [huggingface: skills/hf-cloud-sagemaker-iam-preflight/SKILL.md](../../.agents/skills/huggingface/skills/hf-cloud-sagemaker-iam-preflight/SKILL.md)
- [huggingface: skills/hf-cloud-sagemaker-production-defaults/SKILL.md](../../.agents/skills/huggingface/skills/hf-cloud-sagemaker-production-defaults/SKILL.md)
- [huggingface: skills/hf-cloud-serving-image-selection/SKILL.md](../../.agents/skills/huggingface/skills/hf-cloud-serving-image-selection/SKILL.md)
- [huggingface: skills/hf-mem/SKILL.md](../../.agents/skills/huggingface/skills/hf-mem/SKILL.md)
- [huggingface: skills/huggingface-best/SKILL.md](../../.agents/skills/huggingface/skills/huggingface-best/SKILL.md)
- [huggingface: skills/huggingface-community-evals/SKILL.md](../../.agents/skills/huggingface/skills/huggingface-community-evals/SKILL.md)
- [huggingface: skills/huggingface-datasets/SKILL.md](../../.agents/skills/huggingface/skills/huggingface-datasets/SKILL.md)
- [huggingface: skills/huggingface-gradio/SKILL.md](../../.agents/skills/huggingface/skills/huggingface-gradio/SKILL.md)
- [huggingface: skills/huggingface-llm-trainer/SKILL.md](../../.agents/skills/huggingface/skills/huggingface-llm-trainer/SKILL.md)
- [huggingface: skills/huggingface-local-models/SKILL.md](../../.agents/skills/huggingface/skills/huggingface-local-models/SKILL.md)
- [huggingface: skills/huggingface-lora-space-builder/SKILL.md](../../.agents/skills/huggingface/skills/huggingface-lora-space-builder/SKILL.md)
- [huggingface: skills/huggingface-paper-publisher/SKILL.md](../../.agents/skills/huggingface/skills/huggingface-paper-publisher/SKILL.md)
- [huggingface: skills/huggingface-papers/SKILL.md](../../.agents/skills/huggingface/skills/huggingface-papers/SKILL.md)
- [huggingface: skills/huggingface-spaces/SKILL.md](../../.agents/skills/huggingface/skills/huggingface-spaces/SKILL.md)
- [huggingface: skills/huggingface-tool-builder/SKILL.md](../../.agents/skills/huggingface/skills/huggingface-tool-builder/SKILL.md)
- [huggingface: skills/huggingface-trackio/SKILL.md](../../.agents/skills/huggingface/skills/huggingface-trackio/SKILL.md)
- [huggingface: skills/huggingface-vision-trainer/SKILL.md](../../.agents/skills/huggingface/skills/huggingface-vision-trainer/SKILL.md)
- [huggingface: skills/huggingface-zerogpu/SKILL.md](../../.agents/skills/huggingface/skills/huggingface-zerogpu/SKILL.md)
- [huggingface: skills/train-sentence-transformers/SKILL.md](../../.agents/skills/huggingface/skills/train-sentence-transformers/SKILL.md)
- [huggingface: skills/transformers-js/SKILL.md](../../.agents/skills/huggingface/skills/transformers-js/SKILL.md)
- [huggingface: skills/trl-training/SKILL.md](../../.agents/skills/huggingface/skills/trl-training/SKILL.md)
- [orchestra: 0-autoresearch-skill/SKILL.md](../../.agents/skills/orchestra/0-autoresearch-skill/SKILL.md)
- [orchestra: 01-model-architecture/litgpt/SKILL.md](../../.agents/skills/orchestra/01-model-architecture/litgpt/SKILL.md)
- [orchestra: 01-model-architecture/mamba/SKILL.md](../../.agents/skills/orchestra/01-model-architecture/mamba/SKILL.md)
- [orchestra: 01-model-architecture/nanogpt/SKILL.md](../../.agents/skills/orchestra/01-model-architecture/nanogpt/SKILL.md)
- [orchestra: 01-model-architecture/rwkv/SKILL.md](../../.agents/skills/orchestra/01-model-architecture/rwkv/SKILL.md)
- [orchestra: 01-model-architecture/torchtitan/SKILL.md](../../.agents/skills/orchestra/01-model-architecture/torchtitan/SKILL.md)
- [orchestra: 02-tokenization/huggingface-tokenizers/SKILL.md](../../.agents/skills/orchestra/02-tokenization/huggingface-tokenizers/SKILL.md)
- [orchestra: 02-tokenization/sentencepiece/SKILL.md](../../.agents/skills/orchestra/02-tokenization/sentencepiece/SKILL.md)
- [orchestra: 03-fine-tuning/axolotl/SKILL.md](../../.agents/skills/orchestra/03-fine-tuning/axolotl/SKILL.md)
- [orchestra: 03-fine-tuning/llama-factory/SKILL.md](../../.agents/skills/orchestra/03-fine-tuning/llama-factory/SKILL.md)
- [orchestra: 03-fine-tuning/peft/SKILL.md](../../.agents/skills/orchestra/03-fine-tuning/peft/SKILL.md)
- [orchestra: 03-fine-tuning/unsloth/SKILL.md](../../.agents/skills/orchestra/03-fine-tuning/unsloth/SKILL.md)
- [orchestra: 04-mechanistic-interpretability/nnsight/SKILL.md](../../.agents/skills/orchestra/04-mechanistic-interpretability/nnsight/SKILL.md)
- [orchestra: 04-mechanistic-interpretability/pyvene/SKILL.md](../../.agents/skills/orchestra/04-mechanistic-interpretability/pyvene/SKILL.md)
- [orchestra: 04-mechanistic-interpretability/saelens/SKILL.md](../../.agents/skills/orchestra/04-mechanistic-interpretability/saelens/SKILL.md)
- [orchestra: 04-mechanistic-interpretability/transformer-lens/SKILL.md](../../.agents/skills/orchestra/04-mechanistic-interpretability/transformer-lens/SKILL.md)
- [orchestra: 05-data-processing/nemo-curator/SKILL.md](../../.agents/skills/orchestra/05-data-processing/nemo-curator/SKILL.md)
- [orchestra: 05-data-processing/ray-data/SKILL.md](../../.agents/skills/orchestra/05-data-processing/ray-data/SKILL.md)
- [orchestra: 06-post-training/grpo-rl-training/SKILL.md](../../.agents/skills/orchestra/06-post-training/grpo-rl-training/SKILL.md)
- [orchestra: 06-post-training/miles/SKILL.md](../../.agents/skills/orchestra/06-post-training/miles/SKILL.md)
- [orchestra: 06-post-training/openrlhf/SKILL.md](../../.agents/skills/orchestra/06-post-training/openrlhf/SKILL.md)
- [orchestra: 06-post-training/simpo/SKILL.md](../../.agents/skills/orchestra/06-post-training/simpo/SKILL.md)
- [orchestra: 06-post-training/slime/SKILL.md](../../.agents/skills/orchestra/06-post-training/slime/SKILL.md)
- [orchestra: 06-post-training/torchforge/SKILL.md](../../.agents/skills/orchestra/06-post-training/torchforge/SKILL.md)
- [orchestra: 06-post-training/trl-fine-tuning/SKILL.md](../../.agents/skills/orchestra/06-post-training/trl-fine-tuning/SKILL.md)
- [orchestra: 06-post-training/verl/SKILL.md](../../.agents/skills/orchestra/06-post-training/verl/SKILL.md)
- [orchestra: 07-safety-alignment/constitutional-ai/SKILL.md](../../.agents/skills/orchestra/07-safety-alignment/constitutional-ai/SKILL.md)
- [orchestra: 07-safety-alignment/llamaguard/SKILL.md](../../.agents/skills/orchestra/07-safety-alignment/llamaguard/SKILL.md)
- [orchestra: 07-safety-alignment/nemo-guardrails/SKILL.md](../../.agents/skills/orchestra/07-safety-alignment/nemo-guardrails/SKILL.md)
- [orchestra: 07-safety-alignment/prompt-guard/SKILL.md](../../.agents/skills/orchestra/07-safety-alignment/prompt-guard/SKILL.md)
- [orchestra: 08-distributed-training/accelerate/SKILL.md](../../.agents/skills/orchestra/08-distributed-training/accelerate/SKILL.md)
- [orchestra: 08-distributed-training/deepspeed/SKILL.md](../../.agents/skills/orchestra/08-distributed-training/deepspeed/SKILL.md)
- [orchestra: 08-distributed-training/megatron-core/SKILL.md](../../.agents/skills/orchestra/08-distributed-training/megatron-core/SKILL.md)
- [orchestra: 08-distributed-training/pytorch-fsdp2/SKILL.md](../../.agents/skills/orchestra/08-distributed-training/pytorch-fsdp2/SKILL.md)
- [orchestra: 08-distributed-training/pytorch-lightning/SKILL.md](../../.agents/skills/orchestra/08-distributed-training/pytorch-lightning/SKILL.md)
- [orchestra: 08-distributed-training/ray-train/SKILL.md](../../.agents/skills/orchestra/08-distributed-training/ray-train/SKILL.md)
- [orchestra: 09-infrastructure/lambda-labs/SKILL.md](../../.agents/skills/orchestra/09-infrastructure/lambda-labs/SKILL.md)
- [orchestra: 09-infrastructure/modal/SKILL.md](../../.agents/skills/orchestra/09-infrastructure/modal/SKILL.md)
- [orchestra: 09-infrastructure/skypilot/SKILL.md](../../.agents/skills/orchestra/09-infrastructure/skypilot/SKILL.md)
- [orchestra: 10-optimization/awq/SKILL.md](../../.agents/skills/orchestra/10-optimization/awq/SKILL.md)
- [orchestra: 10-optimization/bitsandbytes/SKILL.md](../../.agents/skills/orchestra/10-optimization/bitsandbytes/SKILL.md)
- [orchestra: 10-optimization/flash-attention/SKILL.md](../../.agents/skills/orchestra/10-optimization/flash-attention/SKILL.md)
- [orchestra: 10-optimization/gguf/SKILL.md](../../.agents/skills/orchestra/10-optimization/gguf/SKILL.md)
- [orchestra: 10-optimization/gptq/SKILL.md](../../.agents/skills/orchestra/10-optimization/gptq/SKILL.md)
- [orchestra: 10-optimization/hqq/SKILL.md](../../.agents/skills/orchestra/10-optimization/hqq/SKILL.md)
- [orchestra: 10-optimization/ml-training-recipes/SKILL.md](../../.agents/skills/orchestra/10-optimization/ml-training-recipes/SKILL.md)
- [orchestra: 11-evaluation/bigcode-evaluation-harness/SKILL.md](../../.agents/skills/orchestra/11-evaluation/bigcode-evaluation-harness/SKILL.md)
- [orchestra: 11-evaluation/lm-evaluation-harness/SKILL.md](../../.agents/skills/orchestra/11-evaluation/lm-evaluation-harness/SKILL.md)
- [orchestra: 11-evaluation/nemo-evaluator/SKILL.md](../../.agents/skills/orchestra/11-evaluation/nemo-evaluator/SKILL.md)
- [orchestra: 12-inference-serving/llama-cpp/SKILL.md](../../.agents/skills/orchestra/12-inference-serving/llama-cpp/SKILL.md)
- [orchestra: 12-inference-serving/sglang/SKILL.md](../../.agents/skills/orchestra/12-inference-serving/sglang/SKILL.md)
- [orchestra: 12-inference-serving/tensorrt-llm/SKILL.md](../../.agents/skills/orchestra/12-inference-serving/tensorrt-llm/SKILL.md)
- [orchestra: 12-inference-serving/vllm/SKILL.md](../../.agents/skills/orchestra/12-inference-serving/vllm/SKILL.md)
- [orchestra: 13-mlops/mlflow/SKILL.md](../../.agents/skills/orchestra/13-mlops/mlflow/SKILL.md)
- [orchestra: 13-mlops/swanlab/SKILL.md](../../.agents/skills/orchestra/13-mlops/swanlab/SKILL.md)
- [orchestra: 13-mlops/tensorboard/SKILL.md](../../.agents/skills/orchestra/13-mlops/tensorboard/SKILL.md)
- [orchestra: 13-mlops/weights-and-biases/SKILL.md](../../.agents/skills/orchestra/13-mlops/weights-and-biases/SKILL.md)
- [orchestra: 14-agents/a-evolve/SKILL.md](../../.agents/skills/orchestra/14-agents/a-evolve/SKILL.md)
- [orchestra: 14-agents/autogpt/SKILL.md](../../.agents/skills/orchestra/14-agents/autogpt/SKILL.md)
- [orchestra: 14-agents/crewai/SKILL.md](../../.agents/skills/orchestra/14-agents/crewai/SKILL.md)
- [orchestra: 14-agents/langchain/SKILL.md](../../.agents/skills/orchestra/14-agents/langchain/SKILL.md)
- [orchestra: 14-agents/llamaindex/SKILL.md](../../.agents/skills/orchestra/14-agents/llamaindex/SKILL.md)
- [orchestra: 15-rag/chroma/SKILL.md](../../.agents/skills/orchestra/15-rag/chroma/SKILL.md)
- [orchestra: 15-rag/faiss/SKILL.md](../../.agents/skills/orchestra/15-rag/faiss/SKILL.md)
- [orchestra: 15-rag/pinecone/SKILL.md](../../.agents/skills/orchestra/15-rag/pinecone/SKILL.md)
- [orchestra: 15-rag/qdrant/SKILL.md](../../.agents/skills/orchestra/15-rag/qdrant/SKILL.md)
- [orchestra: 15-rag/sentence-transformers/SKILL.md](../../.agents/skills/orchestra/15-rag/sentence-transformers/SKILL.md)
- [orchestra: 16-prompt-engineering/dspy/SKILL.md](../../.agents/skills/orchestra/16-prompt-engineering/dspy/SKILL.md)
- [orchestra: 16-prompt-engineering/guidance/SKILL.md](../../.agents/skills/orchestra/16-prompt-engineering/guidance/SKILL.md)
- [orchestra: 16-prompt-engineering/instructor/SKILL.md](../../.agents/skills/orchestra/16-prompt-engineering/instructor/SKILL.md)
- [orchestra: 16-prompt-engineering/outlines/SKILL.md](../../.agents/skills/orchestra/16-prompt-engineering/outlines/SKILL.md)
- [orchestra: 17-observability/langsmith/SKILL.md](../../.agents/skills/orchestra/17-observability/langsmith/SKILL.md)
- [orchestra: 17-observability/phoenix/SKILL.md](../../.agents/skills/orchestra/17-observability/phoenix/SKILL.md)
- [orchestra: 18-multimodal/audiocraft/SKILL.md](../../.agents/skills/orchestra/18-multimodal/audiocraft/SKILL.md)
- [orchestra: 18-multimodal/blip-2/SKILL.md](../../.agents/skills/orchestra/18-multimodal/blip-2/SKILL.md)
- [orchestra: 18-multimodal/clip/SKILL.md](../../.agents/skills/orchestra/18-multimodal/clip/SKILL.md)
- [orchestra: 18-multimodal/cosmos-policy/SKILL.md](../../.agents/skills/orchestra/18-multimodal/cosmos-policy/SKILL.md)
- [orchestra: 18-multimodal/llava/SKILL.md](../../.agents/skills/orchestra/18-multimodal/llava/SKILL.md)
- [orchestra: 18-multimodal/openpi/SKILL.md](../../.agents/skills/orchestra/18-multimodal/openpi/SKILL.md)
- [orchestra: 18-multimodal/openvla-oft/SKILL.md](../../.agents/skills/orchestra/18-multimodal/openvla-oft/SKILL.md)
- [orchestra: 18-multimodal/segment-anything/SKILL.md](../../.agents/skills/orchestra/18-multimodal/segment-anything/SKILL.md)
- [orchestra: 18-multimodal/stable-diffusion/SKILL.md](../../.agents/skills/orchestra/18-multimodal/stable-diffusion/SKILL.md)
- [orchestra: 18-multimodal/whisper/SKILL.md](../../.agents/skills/orchestra/18-multimodal/whisper/SKILL.md)
- [orchestra: 19-emerging-techniques/knowledge-distillation/SKILL.md](../../.agents/skills/orchestra/19-emerging-techniques/knowledge-distillation/SKILL.md)
- [orchestra: 19-emerging-techniques/long-context/SKILL.md](../../.agents/skills/orchestra/19-emerging-techniques/long-context/SKILL.md)
- [orchestra: 19-emerging-techniques/model-merging/SKILL.md](../../.agents/skills/orchestra/19-emerging-techniques/model-merging/SKILL.md)
- [orchestra: 19-emerging-techniques/model-pruning/SKILL.md](../../.agents/skills/orchestra/19-emerging-techniques/model-pruning/SKILL.md)
- [orchestra: 19-emerging-techniques/moe-training/SKILL.md](../../.agents/skills/orchestra/19-emerging-techniques/moe-training/SKILL.md)
- [orchestra: 19-emerging-techniques/speculative-decoding/SKILL.md](../../.agents/skills/orchestra/19-emerging-techniques/speculative-decoding/SKILL.md)
- [orchestra: 20-ml-paper-writing/academic-plotting/SKILL.md](../../.agents/skills/orchestra/20-ml-paper-writing/academic-plotting/SKILL.md)
- [orchestra: 20-ml-paper-writing/ml-paper-writing/SKILL.md](../../.agents/skills/orchestra/20-ml-paper-writing/ml-paper-writing/SKILL.md)
- [orchestra: 20-ml-paper-writing/presenting-conference-talks/SKILL.md](../../.agents/skills/orchestra/20-ml-paper-writing/presenting-conference-talks/SKILL.md)
- [orchestra: 20-ml-paper-writing/systems-paper-writing/SKILL.md](../../.agents/skills/orchestra/20-ml-paper-writing/systems-paper-writing/SKILL.md)
- [orchestra: 21-research-ideation/brainstorming-research-ideas/SKILL.md](../../.agents/skills/orchestra/21-research-ideation/brainstorming-research-ideas/SKILL.md)
- [orchestra: 21-research-ideation/creative-thinking-for-research/SKILL.md](../../.agents/skills/orchestra/21-research-ideation/creative-thinking-for-research/SKILL.md)
- [orchestra: 22-agent-native-research-artifact/compiler/SKILL.md](../../.agents/skills/orchestra/22-agent-native-research-artifact/compiler/SKILL.md)
- [orchestra: 22-agent-native-research-artifact/research-manager/SKILL.md](../../.agents/skills/orchestra/22-agent-native-research-artifact/research-manager/SKILL.md)
- [orchestra: 22-agent-native-research-artifact/rigor-reviewer/SKILL.md](../../.agents/skills/orchestra/22-agent-native-research-artifact/rigor-reviewer/SKILL.md)
- [anthropic: skills/academy-guide/SKILL.md](../../.agents/skills/anthropic/skills/academy-guide/SKILL.md)
- [anthropic: skills/algorithmic-art/SKILL.md](../../.agents/skills/anthropic/skills/algorithmic-art/SKILL.md)
- [anthropic: skills/brand-guidelines/SKILL.md](../../.agents/skills/anthropic/skills/brand-guidelines/SKILL.md)
- [anthropic: skills/canvas-design/SKILL.md](../../.agents/skills/anthropic/skills/canvas-design/SKILL.md)
- [anthropic: skills/claude-api/SKILL.md](../../.agents/skills/anthropic/skills/claude-api/SKILL.md)
- [anthropic: skills/discernment-nudge/SKILL.md](../../.agents/skills/anthropic/skills/discernment-nudge/SKILL.md)
- [anthropic: skills/frontend-design/SKILL.md](../../.agents/skills/anthropic/skills/frontend-design/SKILL.md)
- [anthropic: skills/internal-comms/SKILL.md](../../.agents/skills/anthropic/skills/internal-comms/SKILL.md)
- [anthropic: skills/mcp-builder/SKILL.md](../../.agents/skills/anthropic/skills/mcp-builder/SKILL.md)
- [anthropic: skills/skill-creator/SKILL.md](../../.agents/skills/anthropic/skills/skill-creator/SKILL.md)
- [anthropic: skills/slack-gif-creator/SKILL.md](../../.agents/skills/anthropic/skills/slack-gif-creator/SKILL.md)
- [anthropic: skills/theme-factory/SKILL.md](../../.agents/skills/anthropic/skills/theme-factory/SKILL.md)
- [anthropic: skills/web-artifacts-builder/SKILL.md](../../.agents/skills/anthropic/skills/web-artifacts-builder/SKILL.md)
- [anthropic: skills/webapp-testing/SKILL.md](../../.agents/skills/anthropic/skills/webapp-testing/SKILL.md)
- [openai: skills/.curated/aspnet-core/SKILL.md](../../.agents/skills/openai/skills/.curated/aspnet-core/SKILL.md)
- [openai: skills/.curated/chatgpt-apps/SKILL.md](../../.agents/skills/openai/skills/.curated/chatgpt-apps/SKILL.md)
- [openai: skills/.curated/cli-creator/SKILL.md](../../.agents/skills/openai/skills/.curated/cli-creator/SKILL.md)
- [openai: skills/.curated/cloudflare-deploy/SKILL.md](../../.agents/skills/openai/skills/.curated/cloudflare-deploy/SKILL.md)
- [openai: skills/.curated/define-goal/SKILL.md](../../.agents/skills/openai/skills/.curated/define-goal/SKILL.md)
- [openai: skills/.curated/figma/SKILL.md](../../.agents/skills/openai/skills/.curated/figma/SKILL.md)
- [openai: skills/.curated/figma-code-connect-components/SKILL.md](../../.agents/skills/openai/skills/.curated/figma-code-connect-components/SKILL.md)
- [openai: skills/.curated/figma-create-design-system-rules/SKILL.md](../../.agents/skills/openai/skills/.curated/figma-create-design-system-rules/SKILL.md)
- [openai: skills/.curated/figma-create-new-file/SKILL.md](../../.agents/skills/openai/skills/.curated/figma-create-new-file/SKILL.md)
- [openai: skills/.curated/figma-generate-design/SKILL.md](../../.agents/skills/openai/skills/.curated/figma-generate-design/SKILL.md)
- [openai: skills/.curated/figma-generate-library/SKILL.md](../../.agents/skills/openai/skills/.curated/figma-generate-library/SKILL.md)
- [openai: skills/.curated/figma-implement-design/SKILL.md](../../.agents/skills/openai/skills/.curated/figma-implement-design/SKILL.md)
- [openai: skills/.curated/figma-use/SKILL.md](../../.agents/skills/openai/skills/.curated/figma-use/SKILL.md)
- [openai: skills/.curated/gh-address-comments/SKILL.md](../../.agents/skills/openai/skills/.curated/gh-address-comments/SKILL.md)
- [openai: skills/.curated/gh-fix-ci/SKILL.md](../../.agents/skills/openai/skills/.curated/gh-fix-ci/SKILL.md)
- [openai: skills/.curated/hatch-pet/SKILL.md](../../.agents/skills/openai/skills/.curated/hatch-pet/SKILL.md)
- [openai: skills/.curated/jupyter-notebook/SKILL.md](../../.agents/skills/openai/skills/.curated/jupyter-notebook/SKILL.md)
- [openai: skills/.curated/linear/SKILL.md](../../.agents/skills/openai/skills/.curated/linear/SKILL.md)
- [openai: skills/.curated/migrate-to-codex/SKILL.md](../../.agents/skills/openai/skills/.curated/migrate-to-codex/SKILL.md)
- [openai: skills/.curated/netlify-deploy/SKILL.md](../../.agents/skills/openai/skills/.curated/netlify-deploy/SKILL.md)
- [openai: skills/.curated/notion-knowledge-capture/SKILL.md](../../.agents/skills/openai/skills/.curated/notion-knowledge-capture/SKILL.md)
- [openai: skills/.curated/notion-meeting-intelligence/SKILL.md](../../.agents/skills/openai/skills/.curated/notion-meeting-intelligence/SKILL.md)
- [openai: skills/.curated/notion-research-documentation/SKILL.md](../../.agents/skills/openai/skills/.curated/notion-research-documentation/SKILL.md)
- [openai: skills/.curated/notion-spec-to-implementation/SKILL.md](../../.agents/skills/openai/skills/.curated/notion-spec-to-implementation/SKILL.md)
- [openai: skills/.curated/openai-docs/SKILL.md](../../.agents/skills/openai/skills/.curated/openai-docs/SKILL.md)
- [openai: skills/.curated/pdf/SKILL.md](../../.agents/skills/openai/skills/.curated/pdf/SKILL.md)
- [openai: skills/.curated/playwright/SKILL.md](../../.agents/skills/openai/skills/.curated/playwright/SKILL.md)
- [openai: skills/.curated/playwright-interactive/SKILL.md](../../.agents/skills/openai/skills/.curated/playwright-interactive/SKILL.md)
- [openai: skills/.curated/render-deploy/SKILL.md](../../.agents/skills/openai/skills/.curated/render-deploy/SKILL.md)
- [openai: skills/.curated/screenshot/SKILL.md](../../.agents/skills/openai/skills/.curated/screenshot/SKILL.md)
- [openai: skills/.curated/security-best-practices/SKILL.md](../../.agents/skills/openai/skills/.curated/security-best-practices/SKILL.md)
- [openai: skills/.curated/security-ownership-map/SKILL.md](../../.agents/skills/openai/skills/.curated/security-ownership-map/SKILL.md)
- [openai: skills/.curated/security-threat-model/SKILL.md](../../.agents/skills/openai/skills/.curated/security-threat-model/SKILL.md)
- [openai: skills/.curated/sentry/SKILL.md](../../.agents/skills/openai/skills/.curated/sentry/SKILL.md)
- [openai: skills/.curated/speech/SKILL.md](../../.agents/skills/openai/skills/.curated/speech/SKILL.md)
- [openai: skills/.curated/transcribe/SKILL.md](../../.agents/skills/openai/skills/.curated/transcribe/SKILL.md)
- [openai: skills/.curated/vercel-deploy/SKILL.md](../../.agents/skills/openai/skills/.curated/vercel-deploy/SKILL.md)
- [openai: skills/.curated/winui-app/SKILL.md](../../.agents/skills/openai/skills/.curated/winui-app/SKILL.md)
- [openai: skills/.curated/yeet/SKILL.md](../../.agents/skills/openai/skills/.curated/yeet/SKILL.md)
- [openai: skills/.system/imagegen/SKILL.md](../../.agents/skills/openai/skills/.system/imagegen/SKILL.md)
- [openai: skills/.system/openai-docs/SKILL.md](../../.agents/skills/openai/skills/.system/openai-docs/SKILL.md)
- [openai: skills/.system/plugin-creator/SKILL.md](../../.agents/skills/openai/skills/.system/plugin-creator/SKILL.md)
- [openai: skills/.system/skill-creator/SKILL.md](../../.agents/skills/openai/skills/.system/skill-creator/SKILL.md)
- [openai: skills/.system/skill-installer/SKILL.md](../../.agents/skills/openai/skills/.system/skill-installer/SKILL.md)

## Installation validation

All 408 installed SKILL.md entries and all file hashes in skills.lock.json were
checked against the copied bytes; every catalog path exists and all six excluded
paths are absent. Qvelt checks after integration: 220 tests passed in 4.35s,
Ruff/format passed, strict mypy passed on 21 first-party files. Vendor scripts
were not executed or certified by these checks. Upstream formatting is preserved.

## User-supplied mandatory coding skill

[ponytail](../../.agents/skills/ponytail/SKILL.md) was supplied on 2026-10-31 and
retained byte-for-byte; see lock local_skills/file hash. Its frontmatter declares
MIT. All project AI agents must read and apply it to coding, fixing, reviewing,
refactoring and dependency selection. Default level is full, until the user changes
or disables it. Preserve project-required unit/reference tests, numerical evidence,
validation and reproducibility. Report unchecked scope and material risks at the
end of delivery. The mistakenly supplied trace skill was removed before commit.
Total installed SKILL.md entries: 409.
