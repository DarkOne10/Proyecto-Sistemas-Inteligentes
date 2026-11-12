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

## S4 — Benchmark, vitrina (showcase) y arena en vivo

**Tests:** 140/140

- Benchmark (`sandbox_benchmark`): 6 tareas (`survival_rain`, `endurance`, `mixup`,
  `spiral_evasion`, `collapsing_arena`, `slalom_focus`) × PPO/SAC con evaluación periódica
  contra la línea base aleatoria; escribe `results/benchmark/benchmark_curves.csv`
  (retorno medio, HP medio y tasa de finalización por punto) y, con `--plot`,
  la gráfica `benchmark_curves.png` (extra `benchmark` → matplotlib).
  SAC corre siempre en acción continua y se salta las tareas que exigen foco
  (`_run_kwargs` devuelve `None`); el eje X registra `model.num_timesteps` real.
  Corrección tras verificarla empíricamente: `learn(..., reset_num_timesteps=False)`
  **suma** pasos, así que ahora se le pasa el tramo restante
  (`min(eval_every, timesteps - num_timesteps)`) en vez del objetivo acumulado —
  antes cada corrida disparaba por encima de lo pedido (10 240 pedidos → 14 336 reales).
- `sandbox_training`: `evaluate` se divide en `evaluate` + `evaluate_model` y
  `_make_model`/`_load_model` pasan a públicos (`make_model`, `load_model`) para que
  la vitrina y la arena reutilicen la carga de modelos.
- Vitrina (`python -m toriel_rl.showcase`): recorre las **24 escenas en orden fijo** —
  los 3 ataques originales de Toriel (`fire_helix`, `fire_waves`, `paw_sweep`),
  los 15 patrones (sin `empty`, en orden de registro) y los 6 escenarios —
  con `--seconds` (15) de acción por escena y una tarjeta de título de
  `--title-seconds` (2) antes de empezar, que muestra familia, nombre
  (p. ej. `A3 Blue Collapsing Ring`) y regla de color (p. ej. `Blue: stay still to survive`).
  Política aleatoria por defecto; `--model` acepta `.json` (tabla Q → ataques de Toriel)
  o `.zip` de PPO/SAC (modelo → escenas del sandbox) y usa la aleatoria como respaldo
  donde el modelo no encaja; `--loop` reinicia tras la última escena, `--seed` la hace
  reproducible y cerrar la ventana detiene el recorrido sin perder lo mostrado.
- Arena (`python -m toriel_rl.arena`): escena aleatoria por episodio (con las 24 escenas
  en el sorteo) hasta cerrar la ventana, con `--episodes N` para limitarla y `--random`
  para forzar la política aleatoria. La barra inferior (`draw_arena_hud`) muestra nombre
  de escena, recompensa del episodio, barra de HP e indicador `IFR`.
  Un episodio temprano reinicia la misma escena en la vitrina (preserva el tiempo por escena)
  y en la arena termina y sigue con la siguiente escena.
- Ambos comandos comparten el registro de escenas, las políticas y los overlays en
  `showcase.py` (`Scene`, `build_scenes`, `load_policies`, `detect_algo`, `draw_title_card`,
  `draw_arena_hud`); `arena.py` solo aporta el bucle de episodios aleatorios y su CLI.
  El algoritmo del zip se detecta sin cargarlo (clave `policy_class` serializada) y el
  modo de acción (discreto/foco/continuo) se deriva del `action_space` del modelo.
- Verificación: 140/140 tests (`test_showcase.py`, 15 nuevos, también en verde con pytest);
  demos headless de los 4 caminos — `showcase`, `showcase --model q_table.json`,
  `showcase --model sandbox_ppo.zip`, `arena --random`, `arena --model sandbox_sac.zip`
  y `arena --model q_table.json`; benchmark de control con el fix
  (puntos exactos en 2048 y 4096 para `--timesteps 4096`).

### Resultados del benchmark

Corrida completa `--timesteps 10240 --eval-every 2048 --eval-episodes 3 --seed 7 --plot`:
61 filas en `benchmark_curves.csv` (6 corridas × 5 puntos de evaluación + 6 líneas base
aleatorias; SAC se salta `slalom_focus`), 11 zips por tarea y la gráfica `benchmark_curves.png`.

| Tarea | Base aleatoria | PPO mejor punto | SAC mejor punto | Ganador |
|---|---|---|---|---|
| `collapsing_arena` | −52.0 | −9.9 @4096 | **+20.2 @10240** | SAC |
| `endurance` | −56.5 | −30.2 @8192 | **+8.3 @10240** | SAC |
| `mixup` | −62.9 | −19.7 @2048 | **+9.7 @2048** | SAC |
| `slalom_focus` | −53.2 | **+59.8 @8192** | omitido (exige foco) | PPO |
| `spiral_evasion` | −61.6 | −49.4 @4096 | **−37.9 @6144** | SAC |
| `survival_rain` | −52.0 | −12.7 @8192 | **+20.4 @6144** | SAC |

- **SAC gana 5 de 6 tareas** y PPO se lleva `slalom_focus` (la única que exige
  acción de foco, que SAC no puede representar). En sus mejores puntos ambos
  superan siempre a la línea base aleatoria.
- SAC es el único que cruza a retorno positivo en `endurance`, `mixup`,
  `survival_rain` y `collapsing_arena` (100 % de finalización en esas cuatro);
  PPO solo en `slalom_focus` (+59.8, 100 % de finalización).
- `spiral_evasion` es la tarea más difícil: ambos algoritmos se quedan en
  negativo (mejor SAC −37.9 vs base −61.6), consistente con su clasificación
  como escenario de evasión exigente.
- Curvas no monótonas (p. ej. PPO en `collapsing_arena` pickea a 4096 y cae):
  3 episodios de evaluación por punto ⇒ alta varianza de muestreo; la tendencia
  general es de mejora respecto a la base.
