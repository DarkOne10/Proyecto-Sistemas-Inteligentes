# Agente RL para esquivar ataques de Toriel

Simulación 2D propia inspirada en el combate de Toriel. Incluye Paw Sweep,
Fire Waves, Fire Helix y Faltering; reglas de HP y Spare; agente Q-Learning
tabular; entrenamiento, evaluación y demo con Pygame. No automatiza ni
modifica Undertale.

La guía completa de conceptos, instalación, comandos y resultados está en
[GUIA_EJECUCION.md](GUIA_EJECUCION.md).

## Requisitos e instalación

Se requiere Python 3.10 o posterior y conexión a internet para instalar las
dependencias. Desde la carpeta del proyecto:

```powershell
py -m venv .venv
 .\.venv\Scripts\python.exe -m pip install -e ".[render,rl,test]"
```

En Windows no es necesario activar el entorno virtual. En macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[render,rl,test]"
```

Los comandos de ejecución para Windows están en [GUIA_EJECUCION.md](GUIA_EJECUCION.md).

## Ejecutar

Abrir la simulación con una política aleatoria:

```bash
python -m toriel_rl.main --seed 7
```

Entrenar Q-Learning con 2,000 episodios de práctica aislada (rotando los tres
ataques dañinos), seguidos de 500 combates completos de fine-tuning. El modelo
y los registros se guardan en `results/`:

```bash
python -m toriel_rl.training --episodes 2000 --full-combat-episodes 500 --seed 7
```

Evaluar cada patrón y combates completos contra una política aleatoria usando
las mismas semillas:

```bash
python -m toriel_rl.evaluation --model results/q_table.json --episodes 100
```

Ver la política aprendida en la ventana:

```bash
python -m toriel_rl.main --model results/q_table.json --seed 7
```

Para probar un solo patrón con el modelo, sin jugar las 12 rondas:

```bash
python -m toriel_rl.main --model results/q_table.json --attack fire_waves --attack-only
```

El modelo guardado usa el formato de observación vigente; vuelve a entrenar si
el programa avisa que una tabla Q antigua es incompatible.

## Sandbox Red SOUL (Proyecto Sistemas Inteligentes)

Entorno paralelo `SoulSandboxEnv` (módulos `toriel_rl.sandbox_*`): esquivar balas
codificadas por color dentro de un Bullet Board neutral, sin jefes, historia ni
cambios de color de SOUL. Las pruebas originales de Toriel no se modifican.

Reglas de color: el blanco daña siempre, el azul solo si te mueves sin foco, el
naranja solo si estás quieto y el verde cura; las iFrames evitan daño múltiple.
El flag de movimiento deriva solo de la entrada del agente (el empuje externo de
`external_displacement` —p. ej. el `vortex`— no cuenta como movimiento).

Componentes auto-contenidos que se mezclan libremente:

- **Patrones (16):** `sine_sweeper`, `blue_ring`, `orange_gates`, `rain`,
  `homing_sine`, `dart_volley`, `spiral_barrage`, `drifter`, `pickups`,
  `blue_rain`, `cross_fire`, `ricochet`, `aimed_burst`, `zigzag_rain`,
  `vortex` y `empty`.
- **Modos (5):** `survival` (supervivencia con near-miss), `collector` (cuota de
  pickups verdes), `focus_precision` (tablero pulsante + recompensa de near-miss
  enfocado), `endurance` (45 s con nivel que escala cada 10 s) y `mixup`
  (ruletea patrón y nivel en cada episodio).
- **Escenarios (6):** `vertical_lanes`, `spiral_evasion`, `slalom_switching`,
  `pinball`, `narrow_corridor` y `collapsing_arena` (el tablero se encoge).
- **Acciones:** `discrete` (9 direcciones), `focus` (`MultiDiscrete([9, 2])`,
  movimiento + foco a media velocidad) y `continuous` (`Box(-1, 1, (2,))` con
  zona muerta 0.1 y normalización al disco unitario).

Comandos (en Windows: `.\.venv\Scripts\python.exe -m ...`):

```bash
# demo con política aleatoria
python -m toriel_rl.sandbox_main --seed 7 --mode endurance --pattern cross_fire
python -m toriel_rl.sandbox_main --action-mode continuous --scenario pinball

# entrenar PPO/SAC y comparar contra la política aleatoria
python -m toriel_rl.sandbox_training --timesteps 20000 --mode survival --algo ppo
python -m toriel_rl.sandbox_training --timesteps 10000 --algo sac --pattern drifter

# currículo de 5 etapas (solo PPO)
python -m toriel_rl.sandbox_training --curriculum

# evaluar el modelo PPO incluido (especializado en slalom_switching)
python -m toriel_rl.sandbox_training --eval-only models/slalom_focus/sandbox_ppo.zip --mode focus_precision --scenario slalom_switching --level 3 --eval-episodes 50

# benchmark: curvas de aprendizaje tarea × algoritmo (CSV + PNG opcional)
python -m toriel_rl.sandbox_benchmark --plot

# vitrina: 24 escenas en orden, con política aleatoria
python -m toriel_rl.showcase
# vitrina con el modelo recomendado para mostrar
python -m toriel_rl.showcase --model models/mixup/sandbox_sac.zip
# ataques originales de Toriel con la tabla Q
python -m toriel_rl.showcase --model results/q_table.json

# arena en vivo: escenas aleatorias por episodio hasta cerrar la ventana
python -m toriel_rl.arena --model models/mixup/sandbox_sac.zip
python -m toriel_rl.arena --random
```

El repositorio incluye cinco modelos PPO/SAC especializados:
`models/slalom_focus/sandbox_ppo.zip` (`focus_precision`/`slalom_switching`),
`models/survival_rain/sandbox_ppo.zip` (`survival`/`rain`),
`models/endurance/sandbox_sac.zip` (`endurance`),
`models/mixup/sandbox_sac.zip` (`mixup`) y
`models/collapsing_arena/sandbox_sac.zip` (`survival`/`collapsing_arena`).
Cada modelo se especializa en su tarea; no hay uno general para todas las
escenas. Según la evaluación visual, `mixup` es el modelo recomendado para
mostrar el proyecto; `slalom_focus` sigue siendo la opción especializada para
el escenario `slalom_switching`.

La vitrina (`showcase`) recorre las 24 escenas —los 3 ataques originales de
Toriel, los 15 patrones y los 6 escenarios— con una tarjeta de título de 2 s y
`--seconds` (15) de acción por escena; `--model` acepta un `.json` (tabla Q,
ataques de Toriel) o un `.zip` de PPO/SAC (sandbox) y usa la política aleatoria
en las escenas que el modelo no cubre. La arena (`arena`) elige una escena
aleatoria por episodio y muestra en la barra inferior el nombre de la escena,
la recompensa del episodio, la barra de HP y el indicador `IFR`; corre hasta
cerrar la ventana (`--episodes N` la limita y `--random` fuerza la política
aleatoria). El algoritmo y el modo de acción del zip se detectan solos
(`--algo` permite forzarlos), `--loop` repite la vitrina y `--seed` la hace
reproducible.

Para mostrar las 24 escenas con el modelo `mixup`, ejecuta
`python -m toriel_rl.showcase --model models/mixup/sandbox_sac.zip`. A diferencia
de `arena`, que sortea una escena nueva por episodio, `showcase` las presenta
en orden y termina al final (o vuelve a empezar con `--loop`).

En Windows, desde PowerShell y la carpeta raíz del proyecto:

```powershell
# Recorre las 24 escenas en orden y termina al llegar al final
.\.venv\Scripts\python.exe -m toriel_rl.showcase --model models\mixup\sandbox_sac.zip

# Elige una escena aleatoria por episodio y continúa hasta cerrar la ventana
.\.venv\Scripts\python.exe -m toriel_rl.arena --model models\mixup\sandbox_sac.zip
```

El benchmark escribe `results/benchmark/benchmark_curves.csv` con el retorno
medio, HP medio y tasa de finalización de cada punto (más la línea base
aleatoria) y, con `--plot`, la gráfica `benchmark_curves.png`; la gráfica
requiere `pip install -e ".[benchmark]"`. SAC siempre corre en acción
continua y se salta las tareas que exigen foco. El registro de cada sprint
está en [SPRINTS.md](SPRINTS.md).

## Pruebas

```bash
python -m unittest discover -s tests
```

Los patrones y colisiones son aproximaciones para el proyecto, no una réplica
visual o física exacta del juego. Paw Sweep puede generar una segunda pata en
rondas posteriores; Fire Waves selecciona ocasionalmente su variante difícil.
