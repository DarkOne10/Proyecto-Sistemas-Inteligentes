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

## S3 — Acción continua, SAC, patrones avanzados y escenarios dinámicos

**Tests:** 120/120

- Patrones nuevos (16 en total):
  - `cross_fire` (A6): balas blancas cruzando el tablero desde lados opuestos en filas intercaladas.
  - `ricochet` (A7): balas rápidas anguladas que rebotan en paredes y bordes antes de salir.
  - `aimed_burst` (A8): ráfagas francotiradoras de 3 darts disparadas desde un borde aleatorio, apuntando al SOUL.
  - `zigzag_rain` (A10): lluvia blanca que zigzaguea rebotando entre las paredes laterales.
  - `vortex` (A11): viento rotacional — empuje externo tangencial (protocolo `external_displacement(dt, soul_x, soul_y)`, ahora de 3 argumentos) + balas azules en órbita a radio constante; montar la corriente no cuenta como movimiento, así que el azul es seguro si te quedas quieto.
- Modos nuevos (5 en total):
  - `endurance` (B4): episodio de 45 s; el nivel escala 1→5 cada 10 s mediante el hook `BaseMode.dynamic_level` y cada hita paga +2.0 una sola vez (`extra_reward`).
  - `mixup` (B5): aleatorización de dominio — cada episodio re-ruletea el nivel (1–3) en `reset` y el patrón (excluyendo `empty`/`pickups`) mediante el hook `select_pattern`; el env releer `level = self.mode.level` después de `mode.reset`.
- Escenarios nuevos (6 en total):
  - `pinball` (D4): 4 topes de 16×16 alrededor de la arena, `drifter` por defecto.
  - `narrow_corridor` (D5): pasillo 192×64, `dart_volley` por defecto.
  - `collapsing_arena` (D6): el tablero se encoge hasta la escala 0.7 en 15 s (`board_size` dinámico), `rain` por defecto.
- Acción continua: `Box(-1, 1, (2,))`, zona muerta 0.1 (entrada ≤ 0.1 → quieto, `moving=False`), normalización al disco unitario, sin bit de foco.
- Entrenamiento: `--algo ppo|sac`; SAC exige `action_mode="continuous"` (el CLI lo activa automáticamente), modelo `sandbox_{algo}.zip`, `--curriculum` queda como PPO-only (guard en CLI).
- CLI demo con nuevo `--action-mode`.
- Verificación: 120/120 tests (incluye `check_env` del entorno continuo y smoke de SAC); demos de los 5 patrones, 2 modos, 3 escenarios y acción continua; SAC end-to-end (entrena → evalúa → guarda `sandbox_sac.zip`) y guard `--curriculum --algo sac`.
