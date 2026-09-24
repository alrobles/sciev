# SPRINT → release v0.1 (objetivo: lunes)

## Posicionamiento

**Modelo de decisiones tipadas de propósito general, con especialidad
científica.** En el ecosistema System-One abierto (kev, openjev, laya,
SemIf, djev-spark) somos la **única entrada sobre backbone masked-diffusion
(dLLM)** — y la única con invarianza a permutaciones *exacta*.

## Meta realista vs SOTA

No "vencer a Jev" — eso no es creíble ni necesario. Las claims de v0.1:

| claim | evidencia | estado |
|---|---|---|
| Único decision-model sobre dLLM | arquitectura | ✅ |
| Invarianza a permutaciones exacta (flip=0 construcción, no empirical ~0.03-0.08) | tabla ablación | ✅ |
| K≤4 competitivo con open-SOTA | elite 0.87 / SciFact noul 0.853 / SST-2 0.93 / AGNews 0.854 (kev-9b OOD ~0.81) | ✅ |
| Especialidad científica | SciFact + elite battery + DAPT sobre papers | ✅ / 🔄 |
| Calibración + automation real | ECE + auto@5% en cada eval | ✅ |

Límites honestos (Limitations del paper/model card):
- K>4 débil (Banking77 0.23 — head entrenada en K≤4; fix futuro:
  tournament/chunking de opciones)
- Conocimiento paramétrico acotado por el backbone (GPQA ≈ techo LLaDA)
- Score es el tipo débil de toda la clase (nosotros 0.61, ref. ~0.70)

## Sprint board

### Hoy (sáb)
- [x] DAPT-LoRA corriendo (`dapt_llada_l` 30252052, l40)
- [ ] LICENSE (Apache-2.0, igual que kev/laya)
- [ ] README rewrite → framing de release
- [ ] Paper Tabla 3 con números reales (bench_eval DONE)
- [ ] `sci_llada6` se somete cuando `runs/dapt/lora-final` exista

### Domingo
- [ ] Resultados `da_*` (GPQA/SciFact/elite) → elegir heads del release
      (da_* si ≥ c_*, si no c_* con nota)
- [ ] Model card (arquitectura, tabla, límites, reproducción)
- [ ] Copiar heads a `release/` o subir a GitHub release / HF
- [ ] Tag v0.1

### Lunes
- [ ] Release público en GitHub
- [ ] Decisión nombre final + repo rename si aplica
- [ ] arXiv: enviar si Intro/Discussion quedan; si no, martes

## Decisión de heads para release (criterio)

`da_*` (DAPT) reemplaza a `c_*` si: elite no baja >1pt Y (SciFact sube O
GPQA sube ≥3pts). Si no, release = `c_*` y DAPT queda como experimento
documentado (también es resultado).

## Branding (abierto)

Criterio tipo "jev": corto, pronunciable, no literal. Candidatos vivos:
Decida, y por definir. Se decide antes del tag.
