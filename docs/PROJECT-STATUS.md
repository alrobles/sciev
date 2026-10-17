# PROJECT STATUS — Sciev (open System-One decision model)

**Registro inicial**: 2026-09-20 · **Auditoría de release**: v0.1.1

Los valores verificados y hashes de los 32 reportes están en
`paper/results.json`; sustituyen transcripciones anteriores. v0.1 ya
existe y conserva sus pesos/tag; v0.1.1 mantiene los mismos heads `c_*`.
El repositorio sigue privado. La auditoría también retiró los flips
históricos no canónicos, que comparaban posiciones sin alinear opciones,
y documentó diferencias de entrenamiento que impiden atribuir causalmente
todos los cambios de accuracy al DAPT.

## 1. Dónde vive el proyecto

| recurso | ubicación |
|---|---|
| Repos | release público `github.com/alrobles/sciev` · dev privado `github.com/alrobles/sciev-devel` · local `~/GitHub/sciev` · cluster `/kuhpc/scratch/kbs/a474r867/sciev` |
| Repo hermano (backbone propio) | `github.com/alrobles/ecoreasoner` · cluster `/beegfs/a474r867/ecoreasoner` |
| Datos (cluster) | `data/sci_battery{,_v2}`, `data/bench_external/` |
| Runs/checkpoints | `/kuhpc/scratch/kbs/a474r867/sciev/runs/sci/` |
| Notas internas (gitignored) | `runs/sci/ABLATION.md`, `runs/sci/STRATEGY.md` |
| Paper draft | `paper/main.tex` |
| Reporte cruzado en ecoreasoner | `docs/SCIEV-PARALLEL-WORK.md` |

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
        └─ heads especialistas: choice 67.2M; noul/score 16.8M cada uno
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
| 8 | DAPT-LoRA sobre corpus de papers | ✅ `lora-final` = 5000 steps (≤81.9M token slots; warm start g2000) |
| 9 | Heads `da_*` sobre backbone adaptado + re-eval | ✅ hecho — **DAPT no supera criterio → release = `c_*`** |
| 10 | Baseline ecoreasoner | ✅ `eb_*` sobre bw1_sr — checkpoint g20, control temprano (ver §7) |
| 11 | Paper arXiv | 🔄 `paper/main.tex`: Tablas 3/4 auditadas, PDF verificado |
| 12 | Fase qa-v2 (diagnóstico GPQA + batería paramétrica) | 🔄 en curso — ver §7c |

## 5. Resultados consolidados

### Sci battery elite (target interno)
| modelo | choice | noul | score | flip | auto@5% choice |
|---|---:|---:|---:|---:|---:|
| baseline | 0.775 | 0.707 | 0.522 | retirado | 0.37 |
| **c_\*** | **0.8698** | **0.7825** | **0.6075** | **0.00** | **0.8070** |

El flip del baseline requiere reevaluación con identidades alineadas.

### Benchmarks públicos (reportes archivados, sin nuevos entrenamientos)
| benchmark | tipo | c_* | n |
|---|---|---:|---:|
| GPQA main / diamond | choice | 0.3125 / 0.3182 | 448 / 198 |
| SciFact dev | noul | **0.8500** | 340 |
| SciFact dev | score | 0.6294 | 340 |
| Enron spam (K=2) | choice | 0.7510 | 2000 |
| SST-2 (K=2) | choice | **0.9300** | 872 |
| AG News (K=4) | choice | 0.8537 | 7600 |
| Banking77 (K=77) | choice | 0.2289 | 3080 |

### Hallazgos
1. **Canonical ordering**: fija la presentación de opciones distintas;
   los duplicados tras truncación necesitan una política de desempate.
   Se retira la comparación histórica 0.77→0.00 por el error de alineación.
2. **Heads por tipo**: choice favorece AttnPool multicapa; noul/score MLP.
3. **Receta combinada**: score 0.52→0.61; la comparación por sí sola no
   aísla la contribución de la loss ordinal de los otros cambios.
4. **Volumen ≠ transferencia**: 30× datos off-domain → 0.96 in-dist
   pero −13pts en elite; el volumen por sí solo no bastó en esas corridas.
5. **GPQA sigue siendo débil**: main 0.3125; no demuestra un techo teórico
   ni que la head extraiga toda la información disponible del backbone.
6. **Transferencia zero-shot**: SST-2 0.9300 y AG News 0.8537 sin gradient
   updates sobre esos benchmarks; K=77 es débil (0.2289).

## 6. Decisiones de diseño (y por qué)

| decisión | alternativa rechazada | razón |
|---|---|---|
| Backbone congelado + heads | full-FT | catastrophic forgetting (0.775→0.623) |
| Canonical ordering | order-averaging en train | invarianza exacta vs empírica; gratis |
| Heads por tipo | head única multi-task | cada tipo quiere readout distinto |
| AttnPool multicapa | mean-pool última capa | +7pts choice; conocimiento en capas medias |
| Ordinal aux loss | CE plano / cambiar head | respeta orden 0<1<2 sin tocar decode |
| Eval-only en benchmarks | train en benchmark | sin gradient updates/temp-fit en benchmarks; sí informaron selección |
| LoRA/DAPT | full DAPT | adapter portable (~176MB); memoria depende de GPU/batch |
| No Qwen | — | restricción del usuario; LLaDA/ecoreasoner |
| Jev números internos no publicables | — | MCA 2.3(f); se citan reportes de terceros |

## 7. Estado actual (jobs) — actualizado 2026-09-25

| job | estado | qué produce |
|---|---|---|
| `bench_eval` | ✅ DONE | tabla completa benchmarks públicos |
| `dapt_llada` (30252050) | ⏹ TIMEOUT ~3100/6000 | `runs/dapt/lora-g2000` — 6000 steps (9.5h) nunca cupieron en sixhour |
| `dapt_llada_a/_l` resubmits | ❌ OOM | a40/l40 (48GB): `logits.float()` = [16,1024,126k] fp32 ~8.3GB de pico; además no había resume — habrían reiniciado de cero |
| `dapt_llada_r` (30360611) | ✅ DONE 4h49m | resume g2000→5000 → `runs/dapt/lora-final` (ema ~5.49) |
| `sci_llada6` TAG=dag (30402364, a100) | ✅ DONE 38m | heads `dag_*` sobre lora-g2000 |
| `sci_llada6` TAG=da (30402365, a100) | ✅ DONE 38m | heads `da_*` sobre lora-final |
| `sci_eco` (30360632) | ✅ DONE 29m | baseline `eb_*` sobre bw1_sr (MdLMMoE 1.4B propio) |

### Comparación DAPT vs frozen (elite/GPQA/SciFact acc)

| eval | c_* frozen | dag g2000 | da g5000 |
|---|---:|---:|---:|
| elite choice | 0.8698 | 0.8790 | **0.8836** |
| elite noul | 0.7825 | 0.7790 | 0.7555 |
| elite score | 0.6075 | 0.5978 | 0.5926 |
| GPQA main | 0.3125 | 0.2812 | 0.2589 |
| GPQA diamond | 0.3182 | 0.2677 | 0.2424 |
| SciFact noul | 0.8500 | 0.8118 | 0.7235 |
| SciFact score | 0.6294 | 0.5147 | 0.5000 |
| elite choice auto5 | 0.8070 | 0.8392 | 0.8101 |
| elite noul auto5 | 0.2777 | 0.4125 | 0.3272 |

**Criterio sprint: elite −1pt max Y (SciFact↑ O GPQA +3pt) → NO cumple.**
La selección sigue siendo `c_*`. No es una ablación causal de DAPT:
`c_*` usó head_lr=3e-4, accum=1, warmup=200 y steps=2000/3000/3000;
`dag/da/eb` usaron head_lr=1e-3, accum=8, warmup=100 y 3000 steps por tipo.
La Tabla 4 informa las corridas disponibles con esa salvedad, sin afirmar
que DAPT necesariamente reduzca generalización.

`eb_*` usa bw1_sr checkpoint-g20, de una corrida de solo 20 pasos, con
ventana 768→ctx≤384. No representa un modelo de 1.4B plenamente entrenado.
Elite 0.2802/0.6672/0.3344, GPQA 0.2656/0.2828, SciFact 0.5941/0.3000.
`sft_moe_v2` es incompatible con bw1_sr: targetea módulos HF de LLaDA-MoE-7B.

`dapt.py --resume/--start-step` hace warm start del adapter y avanza datos
/scheduler; no restaura optimizer ni RNG. g5000 inicializó desde g2000
con un optimizer nuevo. El máximo nominal es 81.9M token slots, no un
conteo medido de tokens únicos sin padding. Seleccionar logits antes del
cast reduce memoria, pero no se verificó que DAPT bs16 quepa en 48GB.
`sci_llada6.slurm` acepta `DAPT`/`TAG` por env (`--export=ALL,...`).

## 7b. Fase pareada `systemone-v2` — actualizado 2026-09-26

El screening anterior confundía backbone con hiperparámetros de head y
truncaba contextos en silencio. La fase pareada corrige ambos:

| job | estado | qué produce |
|---|---|---|
| `sci_aligned_v2` (30405852) | ✅ DONE 1m50s | batería `systemone-v2` a max_ctx=960, `--overflow exclude` con razones registradas: elite 1307/3649/3685 train, 300/814/827 dev, 448/1200/1218 eval; GPQA 441+193; SciFact 332×2. 8,576 exclusiones `input_overflow` (p50=783, p95=1858 tokens de pasaje) |
| `sci_llada6` fr_matched (30405856) | ✅ DONE 27m | heads `fr_matched_s7331_*`, DAPT=none |
| `sci_llada6` da_matched (30405857) | ✅ DONE 28m | heads `da_matched_s7331_*` sobre lora-final |
| `sci_controls` fr (30486821) | ✅ DONE 15m | evals empty/shuffle sobre fr_* |
| `sci_controls` da (30486822) | ✅ DONE 17m | evals empty/shuffle sobre da_* |

### Comparación pareada (misma receta scientific-v1, seed 7331, datos v2)

| eval | fr (frozen) | da (g5000) | Δ |
|---|---:|---:|---:|
| elite choice | **0.9799** | 0.9598 | −2.0pt |
| elite noul | **0.9308** | 0.9200 | −1.1 |
| elite score | **0.8957** | 0.7521 | −14.4 |
| GPQA main | **0.2789** | 0.2676 | −1.1 |
| GPQA diamond | 0.3005 | **0.3057** | +0.5 |
| SciFact noul | 0.4639 | **0.7289** | **+26.5** |
| SciFact choice | 0.5000 | **0.5151** | +1.5 |

ECE elite (temp dev-only): fr 0.0167/0.0189/0.0118 vs da 0.0164/0.0163/0.0823.

**Hallazgos:**
1. Evidencia completa importa: elite choice 0.98 vs 0.87 con truncado silencioso.
2. El resultado SciFact se **invierte** respecto del screening: con protocolo
   pareado, DAPT saca al modelo de bajo su majority baseline (0.46→0.73).
   La conclusión anterior era un artefacto de diseño no controlado.
3. El efecto DAPT es dependiente de tarea: ayuda verificación OOD, daña
   scoring ordinal y calibración de score (ECE 0.082 vs 0.012).
4. fr noul en SciFact cae bajo el majority baseline (0.46 vs 0.60): el
   backbone congelado no verifica fuera de dominio; el adaptado sí.

`data/build_evidence_controls.py` genera controles desde
`sci_decisions_eval_text.jsonl` (mismo encoding v2). Las filas de control
llevan `label_status=evidence_control` y `gold_semantics=reference_agreement`:
el `gold` es la posición de la respuesta de referencia original — NO una
nueva verdad sobre la evidencia alterada. empty: 448/1200/1218; shuffle:
447/1146/1172 (101 overflow de donantes largos, registradas).

### Controles de evidencia (agreement-with-reference)

| kind | fr empty | fr shuffle | da empty | da shuffle | azar |
|---|---:|---:|---:|---:|---:|
| choice | 0.875 | 0.873 | 0.806 | 0.785 | ~0.25 |
| noul | 0.810 | 0.669 | 0.807 | 0.667 | ~0.60 |
| score | 0.575 | 0.385 | 0.555 | 0.430 | ~0.34 |

**Hallazgo clave:** choice es ~87% resoluble SIN el pasaje — las opciones
delatan la respuesta (gold = texto del teacher coherente con la pregunta;
distractores = perturbaciones numéricas o respuestas de otro tema). score
sí usa evidencia (cae a azar bajo shuffle); noul cae a su prior de
mayoría. Los números elite pareados son cota superior de capacidad
*grounded*; para afirmar grounding en choice hacen falta distractores
indistinguibles sin pasaje. Paper: `tab:matched` + `tab:controls`.

### Pool estricto `--hard-choice` (data/sci_battery_hard)

Solo same-passage swaps + perturbaciones numéricas (sin cross-passage).
choice eval 448→303 filas (733 registros excluidos honestamente).

| eval | fr | da |
|---|---:|---:|
| hard acc | 0.914 | 0.868 |
| hard empty agree | 0.795 | 0.766 |
| hard shuffle agree | 0.825 | 0.722 |

Reduce leakage ~6-8pt pero no lo elimina: los swaps responden a OTRAS
preguntas → aún distinguibles por coherencia q↔a. Hace falta teacher que
genere distractores de la MISMA pregunta. da cae más bajo evidencia
errónea (0.722 vs 0.825) → filtra mejor la evidencia (coherente con
SciFact +26.5pt).

### Multi-seed (seeds 7331-7333, jobs 30486874-877)

| eval | fr mean±sd | da mean±sd |
|---|---:|---:|
| elite choice | 0.970±.017 | 0.961±.011 |
| elite noul | 0.925±.006 | 0.913±.007 |
| elite score | 0.859±.075 | 0.779±.109 |
| GPQA main | 0.295±.014 | 0.288±.018 |
| GPQA diamond | 0.307±.012 | 0.297±.011 |
| SciFact noul | 0.483±.023 | **0.709±.020** |
| SciFact choice | 0.479±.018 | 0.444±.062 |

La ventaja DAPT en verificación SciFact es robusta (+22.6pt, sd~2);
la pérdida en score es ruidosa (sd hasta ±.11 → el −14.4pt de una
corrida era en parte fluctuación). Paper: `tab:matched` ahora reporta
mean±sd.

### 7c. Fase qa-v2 — diagnóstico GPQA + batería paramétrica (en curso)

Ataca la debilidad conocida (GPQA ~0.30) en dos frentes:

**Phase-A** (`scripts/gpqa_phase_a.slurm`, array 0-7): mide el
conocimiento que ya porta el backbone crudo sobre la cohorte limpia
GPQA (440 main / 193 diamond) — PLL full-mask, leave-one-out verdadero
y label-scoring (`scripts/eval_llada_pll.py`), más generación por
difusión greedy directa-6tok y razonamiento-64tok
(`scripts/eval_llada_prompted.py`). Lectura pre-registrada:
PLL ≈ head → el cuello de botella es el backbone (→ compuerta LoRA);
PLL ≫ head → el readout es la palanca; prompted-64 ≫ PLL → la
generación extrae lo que el scoring no (→ destilar razonamiento).

**Resultados Phase-A** (`runs/sci/phase_a_v2/`, 27-sep — backbone crudo,
sin head, accuracy):

| método | main (n=440) | diamond (n=193) |
|---|---:|---:|
| full-mask PLL | 0.225 | 0.207 |
| leave-one-out PLL | 0.234 | 0.249 |
| label scoring | 0.284 | 0.301 |
| prompted-6 (directo) | 0.259 | 0.249 |
| prompted-64 (razonamiento) | 0.250 | 0.228 |
| head v0.2 (referencia) | 0.295 | 0.307 |
| head qa-v2 (§7c) | 0.273 | 0.269 |

**Veredicto**: el mejor readout backbone-only (label, 0.284/0.301) ≈
head (0.295/0.307) — el backbone no esconde conocimiento GPQA que el
readout desperdicie; la head de hecho exprime *más* señal que el
scoring directo. prompted-64 ≈ azar (0.25) — la generación tampoco
extrae nada. Y la head qa-v2 entrenada sobre ~50k MCQA no movió GPQA
(0.27) aunque domina su dominio (qa_eval 0.838, sci_elite 0.920 —
`runs/sci/eval_qa_v2_s7331/`). GPQA es un **capability ceiling de
LLaDA-8B**, no un defecto del decision layer — resultado negativo
limpio que refuerza el framing "evidence-grounded" del producto.

**Consecuencia para la compuerta LoRA**: el criterio <0.38 dispararía,
pero fue diseñada para el mundo "readout bottleneck". Con Phase-A
cerrado, LoRA sobre datos MCQA no inyecta conocimiento graduado —
**no disparar** salvo que se reformule como DAPT sobre corpus de
texto científico (inyección de conocimiento, techo esperado aun así
bajo para preguntas google-proof). El lever real para GPQA es
evidencia en el `state` (GPQA+RAG = el caso de uso grounded del
producto).

**Batería qa_choice v3** (`data/build_qa_choice.py`,
`scripts/qa_build.slurm` → `data/qa_battery_v3`): ARC, OpenBookQA, SciQ,
MMLU-aux (20k), MedMCQA (20k train + 1k dev). Partición global por
grupo de pregunta con prioridad eval > dev > train; pantalla GPQA sin
etiquetas (Jaccard ≥ 0.6 sobre los CSVs main/diamond/extended, no solo
la cohorte limpia); grupos con golds conflictivos eliminados enteros;
dev repartido a roles select/calibration/policy por hash de grupo.
Auditoría de disjointness y digests: `scripts/audit_qa_dataset.py`.

**Train** (`sciev/qa_train.py`, `qa_v2_train.slurm`): head AttnPool
spanpool choice; GPQA vetado de train y selección
(`require_training_data`); resume exacto (optimizer + RNG + order);
mejor checkpoint por (acc, −nll) sobre subset de selección
determinista; modo `--smoke` verifica que `decision.pt` recarga a las
mismas probabilidades.

**Compuerta LoRA** (`scripts/qa_lora_gate.py`): criterio predefinido —
solo si TODAS las acc frozen-head en GPQA-main (cohorte limpia, n=440)
quedan bajo 0.38 se somete `qa_v2_train.slurm` con LORA_RANK=8,
4000 steps, init desde la head frozen, seguido de `qa_v2_eval.slurm`
con dependencia `afterok`. GPQA informa esta decisión → los números
GPQA resultantes son selection-informed (ya declarado en MODEL_CARD).

**Eval** (`sciev/qa_eval.py`, `qa_v2_eval.slurm`): temperatura en el
split calibration, política de aceptación en el split policy
(disjuntos), evaluación en gpqa main/diamond + qa_eval + sci_elite con
chequeos checkpoint/selection/evaluation (`protocol_version` 2);
rechaza checkpoints con presupuesto de entrenamiento incompleto.

**Ensemble** (`scripts/ensemble_instr_variants.py` +
`gpqa_build_variants`/`gpqa_ensemble_v2`/`gpqa_aggregate_v2.slurm`):
variantes de instrucción unidas por identidad
(split_group + option_keys), alineación por contenido (no posición),
CI bootstrap por grupo de pregunta, política de aceptación ajustada en
dev-evals disjuntas.

**Brazo CLM-style** (`sciev/biencoder.py`, `scripts/bi_v2.slurm`):
apareció CLM-8B (Contrastive-LM, Apache-2.0) — otro System-One abierto
con API TypeSafe-compatible: bi-encoder contrastivo (InfoNCE state×
action) sobre Qwen3-8B congelado, 60M pares Nemotron + 30M hard
negatives sintéticos + 1M trayectorias agénticas. Sus pesos requieren
el encoder Qwen3-8B → **fuera de alcance** (restricción del usuario:
nada de pesos Qwen). En su lugar implementamos su ARQUITECTURA como
brazo controlado sobre LLaDA: mismos datos qa-v2, mismas cohortes → la
comparación joint-encoding vs dot-product queda pareada por backbone y
datos (el confound Qwen-vs-LLaDA y 60M-vs-propios se declara, no se
mezcla). Embeddings cacheados por content-hash (backbone congelado →
head training sin forwards de backbone); InfoNCE bidireccional con
distractores same-question como hard negatives + in-batch. Eval bajo
protocolo v2 con temperatura en calibration y política en policy.
Además `serve.py` gana paridad de wire-format: `temperature` por
request (compone con la calibrada), `POST /v1/rank` (best-of-N sobre
candidatos verbatim, implementado como decisión choice) y `GET
/health`.

**Resultado del brazo** (`runs/sci/eval_bi_v2_s7331/`, job 30929246 —
embed 14 shards ~250k vectores, train 4000 pasos en 53s sobre vectores
cacheados, selección plateau 0.371@step2500): el dot-product pierde
feamente contra el encoding conjunto a nuestra escala de datos:

| suite | bi-encoder | head conjunta |
|---|---:|---:|
| qa_eval | 0.376 | 0.838 |
| sci_elite | 0.263 | 0.920 |
| gpqa_main | 0.241 | 0.295 |
| gpqa_diamond | 0.244 | 0.307 |

Lectura: CLM necesitó 60M pares + 30M hard negatives + 1M trayectorias
para que el bi-encoder funcione — es una arquitectura data-hungry; a
~50k decisiones, el state y la opción que nunca se atienden entre sí
no alcanzan. Valida el joint encoding como diseño, no solo como
elección. Una configuración (layer=-1 mean-pool, proj 512): el claim
es "a escala pareada", no "bi-encoders no sirven".

**Fase-B1 — prefiltro K>4 descartado con datos** (`runs/sci/
recall_bi_v2_s7331/`, job 30996734): la idea de reusar el cache como
shortlist barata se midió y no sobrevive. En banking77 (3080 filas ×
77 labels — el test bed K>4 real) el gold cae en el top-k del
dot-product:

| k | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| recall@k | 0.019 | 0.089 | 0.157 | 0.256 | 0.465 |

Un prefiltro top-4 perdería el gold el 91% de las veces (se necesita
recall@4 ≳ 0.85); ni siquiera top-32 cubre la mitad. El brazo se
entrenó en decisiones de 4 opciones — nunca aprendió a separar 77
intents finos. En retrieval global (pool=suite, 746–15,460 acciones
únicas) recall@64 = 0.15–0.40: tampoco sirve como retriever general.
La vía K>4 que queda es chunked-tournament con la propia head
conjunta, no prefiltrado por embeddings.

**Fase-B2 — GPQA grounded** (`runs/sci/gpqa_grounded_v2/`, job
31029331): variantes pareadas del cohort limpio — mismas opciones y
orden (byte-idénticas), solo `ctx` gana un bloque `Evidence:`.
`oracle` = explicación experta del benchmark (cota superior revela-
gold); `nev` = primera traza no-vacía `Explanation_NEV_{1,2,3}` de un
validator no-experto (ruidosa; ~1/3 aciertan). Builder:
`data/build_gpqa_grounded.py` →
`data/bench_external/gpqa/systemone-v2/gpqa_{main,diamond}_choice_eval_{oracle,nev}.jsonl`
(440/440 y 193/193 filas, ≤5 ctx-truncados a max_ctx=896).

| evidencia | c_choice main | c_choice diamond | qa_v2 main | qa_v2 diamond |
|---|---:|---:|---:|---:|
| closed-book | 0.295 | 0.307 | 0.273 | 0.269 |
| nev | 0.318 | 0.332 | 0.284 | 0.269 |
| oracle | **0.695** | 0.326 | **0.707** | 0.244 |

Lectura: (1) la head conjunta SÍ explota evidencia — oracle duplica
main y ambas heads convergen (0.695≈0.707), así que el uso de
evidencia vive en la arquitectura, no en el fine-tune; (2) nev
recupera casi nada → el gap oracle−nev tasa el costo del ruido: el
retrieval útil debe ser de grado explicativo; (3) diamond resiste
hasta al oracle → en el subconjunto más duro el residual es
discriminación de opciones, no conocimiento. El lever GPQA real es
retrieval de calidad alimentando `state` — eso es exactamente el
producto grounded de sciev.

**Posicionamiento** (revisado: pertenencia, no competencia): el
objetivo no es rivalizar con CLM sino mantener a sciev dentro del
ecosistema System-One que CLM acaba de solidificar — mismo contrato
wire (`/v1/systemone`, `/v1/rank`) para que harnesses y clientes
TypeSafe-style nos alcancen igual. Nuestro sitio en el ecosistema: el
miembro masked-diffusion sobre backbone no-Qwen (útil donde las listas
de modelos aprobados excluyen Qwen3-8B) con encoding conjunto
state×option (README §ecosystem). Alcance honesto declarado: CLM
conserva ventaja de latencia (cache de embeddings, revisita sub-ms) y
K irrestricto; nuestro release sigue ≤4 opciones. El brazo pareado
documenta la comparación arquitectónica *dentro* del ecosistema.

### 7d. Fase C2 — anti-leakage retrain (noul/score)

Los controles de evidencia (§7b) midieron el leak pero no lo corrigen:
con evidencia destruida, noul responde desde su prior de mayoría
(empty 0.81 / shuffle 0.67 vs azar ~0.60) y score retiene acuerdo
parcial. `data/build_antileak_train.py` reutiliza
`make_evidence_controls` pero remapea el gold a rechazo
(noul→índice 1 = "false"/no; score→índice 0 = nivel más bajo de la
rúbrica) y etiqueta las filas `label_status=antileak_train`,
conservando el marcador `evidence_control` y los campos
`reference_*` como provenance. Choice queda fuera de alcance: una
fila choice con evidencia corrupta no tiene gold de rechazo válido —
el leak de opciones es un finding de evaluación, no supervisión
entrenable (la vía es teacher de distractores same-question).

`scripts/antileak_v2.slurm` construye los controles desde
`sci_decisions_train_text.jsonl`, los mezcla 1:1:1 con el train
limpio (`sci_{kind}_train.jsonl` + empty + shuffle), re-entrena
`c_noul_al`/`c_score_al` con la receta de release `c_*` (mlp
spanpool, canonical, freeze, 3000 pasos, head_lr 3e-4, warmup 200;
score + `--ordinal 1.0`), y evalúa batería limpia + ambos controles
con `--train-reference` apuntando al mix real (audit de
disjointness) y temp-fit en dev limpio. Criterio de éxito:
el agreement-with-reference en los controles cae hacia la base
rate de la clase de rechazo entre los labels de referencia (si la
head siempre rechaza evidencia corrupta, solo coincide con la
referencia cuando ésta ya era la clase de rechazo — ~0.667 para
noul en esta cohorte), sin colapsar la batería limpia.

**Resultados noul** (`runs/sci/antileak_v2/`, jobs 31043205 +
31044968; el segundo resubió tras el fix `criteria=None` y murió
por time-limit a mitad del primer eval de score — resubmit
31053405 corre los 3 evals de score con skip-guards):

| eval | fr (§7b) | c_noul_al |
|---|---:|---:|
| clean noul | 0.9308 | 0.8958 |
| ctl_empty agree | 0.810 | 0.6667 |
| ctl_shuffle agree | 0.669 | 0.6667 |

Lectura: agreement cae a exactamente la base rate "false" (0.667)
en ambos modos — firma de rechazo uniforme de evidencia corrupta,
no de prior residual (un colapso a mayoría habría arrastrado clean
a ~0.6; se mantiene 0.896). El costo es −3.5pt en clean
(sobre-rechazo parcial o varianza de seed única). ECE alto en
controles (~0.32) es artefacto esperado: la head rechaza
confiada lo que el archivo etiqueta por referencia. Los JSONs de
noul quedaron sin `train_overlap_check` (corrieron antes del flag);
los de score lo traerán. `c_noul_al_stale/` conserva el head
pre-fix para auditoría.

**Resultados score** (resubmit 31053405, completado 2026-10-08):

| eval | fr (§7b) | c_score_al |
|---|---:|---:|
| clean score | 0.8957 | 0.7553 |
| ctl_empty agree | — | 0.3358 |
| ctl_shuffle agree | — | 0.3345 |

Misma firma que noul: agreement colapsa a la base rate del label de
rechazo (~1/3 para rubric de 3 niveles; la referencia es índice 0 =
nivel más bajo) — rechazo uniforme de evidencia corrupta en ambos
modos. Costo mayor que en noul: −14pt en clean (vs −3.5pt), consistente
con que score tiene más vías de sobre-rechazo (cualquier nivel alto
reclasificado a 0). Los JSONs de score sí traen `train_overlap_check`.

Estado 2026-10-08: ambos kinds completos. La decisión de si
`c_*_al` reemplazan a `c_*`/`fr_*` para release queda abierta —
grounding perfecto en controles a costa de clean (−3.5pt noul,
−14pt score).

## 8. Siguiente

1. Heads seleccionados: **`c_*`**, sin cambios respecto de v0.1.
2. Tabla 3 (benchmarks) y Tabla 4 (candidatos) verificadas contra los JSON;
   PDF compilado y regresiones de tablas/flip incluidas en tests.
3. Publicar v0.1.1 sin mover v0.1 ni cambiar la visibilidad privada:
   paper, resultados/provenance, checksums y los mismos tres heads.
4. Antes de un envío arXiv: revisión científica y bibliográfica; si se
   quiere atribuir causalidad a DAPT, repetir con head-training igualado.
5. Reevaluar flips no canónicos con la métrica corregida y reservar un
   test final independiente de la selección de candidatos.

**Actualización 2026-10-02**: v0.2.3 publicado (mismos pesos `fr_*`;
README/MODEL_CARD/manifests alineados). La línea activa es la fase
qa-v2 (§7c): Phase-A diagnostics → `qa_build` → `qa_v2_train` frozen →
compuerta LoRA → `qa_v2_eval`; ensemble de variantes de instrucción
como segundo brazo. CI de pytest en `.github/workflows/test.yml`
(corre la suite completa en CPU, ~13s). Artefactos ya des-trackeados
(`dist/`, `paper/main.{aux,log,out}`, `evals/*.log` en .gitignore).

**Actualización 2026-10-16**: v0.3.0 — release anti-leakage. Heads
servidos: `alp25_{noul,score}` (retrain ρ=0.25 sobre la receta `fr`;
controls a prior de rechazo, SciFact canónica 0.464→0.702) +
`fr_choice` sin cambios. Temps 1.403/0.7641 fit en dev. La decisión de
choice anti-leakage queda para el siguiente release: `alp25x`
(counterfactual) ya supera a `fr` en GPQA-oracle (0.725 vs 0.695) pero
compromete más en evidencia destruida (0.40–0.45 vs 0.26–0.32 de
`alp25p`); trade-off documentado en `tab:antileak-choice`.

