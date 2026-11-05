# Registro de sprints — Roadmap Red SOUL sandbox

## S1 — Patrones base, modo supervivencia y harness PPO

**Commit:** `5f06a9e` · **Tests:** 58/58

- Patrones: `sine_sweeper` (A1), `blue_ring` (A3), `orange_gates` (A4) + utilitario `rain`.
- Modo `survival` (B1): 30 s, recompensa por near-miss (+0.1), lluvia con rampa 1→6 balas/s.
- Escenario `vertical_lanes` (D1): carril seguro que rota en un temporizador.
- Harness PPO (`[rl]` → stable-baselines3 2.9.2): `sandbox_training` con train/evaluate/evaluate_random y CSV de episodios.
- Visualización con tablero centrado, obstáculos y flags MOV/IFR.
- Verificación: demo headless OK; PPO con 4096 pasos vence al aleatorio (retorno −16.6 vs −24.9; HP 8.0 vs 2.7).

## S2 — Acción de foco, modos de objetivo, escenarios de cambio y currículo

**Tests:** 91/91

- Protocolo de patrones ampliado a `update(dt, soul_x, soul_y)` (necesario para el homing; igual que los ataques de Toriel).
- Patrones nuevos: `homing_sine` (A2), `dart_volley` (A9), `spiral_barrage` (A5) + utilitarios `drifter`, `pickups`, `blue_rain`.
- Modos nuevos:
  - `collector` (B2): cuota de greens (`4 + nivel`), 2 obstáculos propios, bono de objetivo +20.
  - `focus_precision` (B3): tablero pulsante (±8 %, período 4 s), recompensa de near-miss enfocado (+0.15).
- Espacio de acción `focus`: `MultiDiscrete([9, 2])` = movimiento + foco; el foco reduce la velocidad ×0.5.
  Regla de color ampliada: el azul daña solo si se mueve **sin foco** (`damages(color, moving, invulnerable, focused)`); el flag `is_moving` sigue derivado solo de la entrada del agente.
- Escenarios nuevos: `spiral_evasion` (D2, tablero 176×176 con `spiral_barrage` por defecto) y
  `slalom_switching` (D3, 192×128, filas de muros naranja/azul alternos con hueco lateral alternante que exigen alternar mover/quieto).
- Métricas de episodio en `info` y en el CSV: `dodged`, `focus_steps`, `wall_hits`, `damage_by_color`, `objective`.
- Obstáculos del entorno = concatenación de modo + escenario.
- `curriculum.py`: 5 etapas (`intro_drifter` → `sine_dodge` → `gates_rain` → `spiral_evasion` → `slalom_focus`),
  `mask_k=3` en las etapas 1–2, criterios de promoción (retorno medio y tasa de finalización) y CLI `--curriculum` con reintentos por etapa.
- Verificación: 91/91 tests; demos de los 4 modos/escenarios nuevos; PPO con acción focus: retorno +34.7 vs −48.1 del aleatorio (completion 100 % vs 0 %).
