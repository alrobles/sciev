# PROJECT STATUS — reverse-jev (open System-One decision model)

**Fecha**: 2026-09-20 · **Autor**: Devin (sesión de desarrollo)

## 1. Dónde vive el proyecto

| recurso | ubicación |
|---|---|
| Repo principal | `github.com/alrobles/sciev-devel` · local `~/GitHub/reverse-jev` · cluster `/beegfs/a474r867/reverse-jev` |
| Repo hermano (backbone propio) | `github.com/alrobles/ecoreasoner` · cluster `/beegfs/a474r867/ecoreasoner` |
| Datos (cluster) | `data/sci_battery{,_v2}`, `data/bench_external/` |
| Runs/checkpoints | `/beegfs/a474r867/reverse-jev/runs/sci/` |
| Notas internas (gitignored) | `runs/sci/ABLATION.md`, `runs/sci/STRATEGY.md` |
| Paper draft | `paper/main.tex` |
| Reporte cruzado en ecoreasoner | `docs/REVERSE-JEV-PARALLEL-WORK.md` |

## 2. Qué se planteó

Construir un equivalente abierto y entrenable de la clase "System One"
(Jev, TypeSafe AI): estado no estructurado + preguntas tipadas →
decisiones estructuradas con probabilidades calibradas, sin generación.

Requisitos: tipos `choice`/`noul`/`score`, probabilidades + calibración +
automation, evaluable en decisiones científicas reales, comparable con
reportes públicos, datos y modelos legalmente usables.

## 3. Arquitectura final (lo que se construyó)

```
LLaDA-8B-Instruct (congelado, bidireccional por construcción)
   └─ hidden states de capas {8,16,24,32}
        └─ heads especialistas por tipo (~2M params)
             choice: AttnPoolHead (query → pooling sobre span de opción)
             noul:   MLP (mean-pool)
             score:  MLP + loss ordinal CORAL (P(y≥j) por BCE)
   + canonical ordering: opciones ordenadas por contenido →
     invarianza a permutaciones exacta (flip = 0 por construcción)
   + temp-fit en dev por tipo
   + (--lora-adapter): backbone adaptable vía DAPT-LoRA
```

## 4. Roadmap y milestones

| fase | contenido | estado |
|---|---|---|
| 0 | Investigación Jev/System One + scaffold repo | ✅ |
| 1 | R1 token-slot readout + pipeline eval | ✅ |
| 2 | R2 marker/spanpool heads, calibración, temp-fit | ✅ |
| 3 | Ablación readouts (mlp/attnpool × orders) | ✅ — attnpool gana choice, orders-train falla |
| 4 | Heads por tipo + canonical ordering + ordinal | ✅ — `c_*`: 0.870/0.783/0.608, flip 0 |
| 5 | Fase B datos (sci_battery_v2, 30×) | ✅ — descubrió domain-shift |
| 6 | Benchmarks externos (GPQA, SciFact, clasificación) | ✅ |
| 7 | Contraste bidireccional con Jev | ✅ — interno completo |
| 8 | DAPT-LoRA sobre corpus de papers | ✅ `lora-final` = 5000 steps (~82M tokens, resume g2000) |
| 9 | Heads `da_*` sobre backbone adaptado + re-eval | 🔄 `dag_*` (g2000) + `da_*` (lora-final) encolados |
| 10 | Baseline ecoreasoner | ✅ `eb_*` sobre bw1_sr — backbone en azar (ver §5) |
| 11 | Paper arXiv | 🔄 `paper/main.tex` skeleton compilable |

## 5. Resultados consolidados

### Sci battery elite (target interno)
| modelo | choice | noul | score | flip | auto@5% choice |
|---|---:|---:|---:|---:|---:|
| baseline | 0.775 | 0.707 | 0.522 | 0.76 | 0.37 |
| **c_\*** | **0.870** | **0.783** | **0.608** | **0.00** | **0.81** |

### Benchmarks públicos (eval-only)
| benchmark | tipo | open (c_*) | ref. publicada |
|---|---|---:|---:|
| GPQA main / diamond | choice | 0.315 / 0.328 | LLaDA techo ~0.33 |
| SciFact dev | noul | **0.853** | ~0.89 (SOTA-decision) |
| SciFact dev | score | 0.624 | ~0.70 |
| Enron spam (K=2) | choice | 0.752 | 0.987 |
| **SST-2 (K=2)** | choice | **0.930** | 0.957 |
| AG News (K=4) | choice | 0.854 | 0.913 |
| Banking77 (K=77) | choice | 0.229 | 0.760 |

### Hallazgos
1. **Canonical ordering**: flip 0.77→0.00 exacto y gratis; el promedio
   de logits en train NO lo lograba (y costaba accuracy).
2. **Heads por tipo**: choice quiere AttnPool multicapa; noul/score MLP.
3. **Ordinal loss**: score 0.52→0.61 (el tipo débil de todos los
   sistemas, incluido el propietario).
4. **Volumen ≠ transferencia**: 30× datos off-domain → 0.96 in-dist
   pero −13pts en elite. Cobertura del tipo de razonamiento > volumen.
5. **La head ya extrae todo el backbone**: GPQA 0.315 ≈ techo LLaDA
   0.33 — el gap de conocimiento es del backbone, no del readout.
6. **Transferencia zero-shot sorprendente en K≤4**: SST-2 0.930 y
   AG News 0.854 sin entrenar en clasificación; colapsa en K=77
   (0.229 — la head nunca vio >4 opciones).

## 6. Decisiones de diseño (y por qué)

| decisión | alternativa rechazada | razón |
|---|---|---|
| Backbone congelado + heads | full-FT | catastrophic forgetting (0.775→0.623) |
| Canonical ordering | order-averaging en train | invarianza exacta vs empírica; gratis |
| Heads por tipo | head única multi-task | cada tipo quiere readout distinto |
| AttnPool multicapa | mean-pool última capa | +7pts choice; conocimiento en capas medias |
| Ordinal aux loss | CE plano / cambiar head | respeta orden 0<1<2 sin tocar decode |
| Eval-only en benchmarks | train en benchmark | cero leakage, contraste limpio |
| LoRA/DAPT | full DAPT | cabe en 1 GPU, adapter portable (~300MB) |
| No Qwen | — | restricción del usuario; LLaDA/ecoreasoner |
| Jev números internos no publicables | — | MCA 2.3(f); se citan reportes de terceros |

## 7. Estado actual (jobs) — actualizado 2026-09-25

| job | estado | qué produce |
|---|---|---|
| `bench_eval` | ✅ DONE | tabla completa benchmarks públicos |
| `dapt_llada` (30252050) | ⏹ TIMEOUT ~3100/6000 | `runs/dapt/lora-g2000` — 6000 steps (9.5h) nunca cupieron en sixhour |
| `dapt_llada_a/_l` resubmits | ❌ OOM | a40/l40 (48GB): `logits.float()` = [16,1024,126k] fp32 ~8.3GB de pico; además no había resume — habrían reiniciado de cero |
| `dapt_llada_r` (30360611) | ✅ DONE 4h49m | resume g2000→5000 → `runs/dapt/lora-final` (ema ~5.49) |
| `sci_llada6` TAG=dag (30360612) | ⏳ PD pro6000 | heads `dag_*` sobre **lora-g2000** — señal DAPT temprana |
| `sci_llada6` TAG=da (30402330) | ⏳ PD pro6000 | heads `da_*` sobre **lora-final** — candidato release |
| `sci_eco` (30360632) | ✅ DONE 29m | baseline `eb_*` sobre bw1_sr (MdLMMoE 1.4B propio) |

Baseline eb_* (bw1_sr, ventana 768→ctx≤384): elite 0.280/0.667/0.334,
GPQA 0.266/0.283, SciFact 0.594/0.300 — choice y score en el azar;
el conocimiento de LLaDA-8B es lo que carga a c_*. NOTA: `sft_moe_v2`
NO se apila sobre bw1_sr (su lora.pt targetea HF LLaDA-MoE-7B
`model.layers.*`, otra arquitectura; `models/` vacío en el cluster).

Cambios 25-sep: `dapt.py` ganó `--resume`/`--start-step` (PeftModel.from_pretrained
is_trainable + fast-forward de scheduler y corpus) y el CE ahora hace
masked-select en bf16 antes del cast fp32 (~8 GiB menos de pico → cabe en
48GB). `sci_llada6.slurm` acepta `DAPT`/`TAG` por env (`--export=ALL,...`).

## 8. Siguiente

1. Cuando `dag_*` evals caigan → comparar vs `c_*` (criterio: elite no baja
   >1pt Y SciFact sube O GPQA +3pts) — decide si g2000 ya basta
2. Cuando `lora-final` (g5000) caiga → `sbatch scripts/sci_llada6.slurm`
   (TAG=da default) → decisión final de heads para release
3. Baseline heads sobre ecoreasoner `bw1_sr`+`sft_moe_v2` (16 capas →
   `--r2-layers=-1,-5,-9,-13`)
4. Tag v0.1 + release público GitHub (repo ya renombrado `sciev-devel`,
   model card + README listos)
5. arXiv: paper tiene intro/battery/discussion; falta Tabla 3 con da_*
6. Decidir venue post-arXiv (workshop ML vs MEE eco)
