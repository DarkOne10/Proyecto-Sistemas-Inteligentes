Proyecto: Agente de aprendizaje por refuerzo para esquivar a Toriel
1. Objetivo y alcance
Construir en Python una simulación 2D inspirada en la batalla contra
Toriel de Undertale. Un agente controla el SOUL dentro del Bullet
Board y aprende a evitar ataques mediante aprendizaje por refuerzo.

Alcance inicial: reproducir las mecánicas relevantes de los cuatro
ataques descritos, no el juego completo. Se recomienda una simulación
propia con Pygame, sin automatizar ni modificar el ejecutable original.
2. Ataques que se deben implementar
Los daños descritos en la fuente asumen que el protagonista no tiene
armadura y está en LV 1.
2.1 Paw Sweep --- barrido de pata
Mecánica descrita - Una pata aparece en la esquina superior
izquierda del Bullet Board y barre en arco hasta la esquina superior
derecha. - Deja un rastro de bolas de fuego. - Cuando la pata llega al
otro lado, las bolas se lanzan hacia el SOUL y rebotan contra las
paredes. - En rondas posteriores aparece una segunda pata desde la
esquina inferior derecha, reflejando el movimiento de la primera. - El
ataque termina inmediatamente si el SOUL toca cualquiera de las patas.

Daño: 3 HP por impacto de bola de fuego; 4 HP por contacto con una
pata.

Implementación propuesta 1. Crear una entidad Paw con posición,
tamaño, dirección y progreso del barrido. 2. Moverla por una trayectoria
curva o aproximada mediante segmentos. 3. Generar bolas de fuego a lo
largo del recorrido. 4. Al terminar el barrido, activar el movimiento de
las bolas hacia el SOUL. 5. Implementar rebotes en los límites del
tablero. 6. Detectar por separado colisiones SOUL--pata y SOUL--bola. 7.
Añadir la segunda pata como variante posterior.

Prioridad: alta; comenzar con una sola pata.
2.2 Fire Waves --- olas de fuego
Mecánica descrita - Corrientes de bolas de fuego caen desde la parte
superior formando un patrón entrecruzado de doble hélice. - Las bolas se
acumulan en la parte inferior. - Hay huecos en las corrientes para que
el SOUL pueda esquivar, además de dos zonas seguras en las esquinas
superiores. - En la versión difícil el Bullet Board es más corto y hay
bolas suspendidas a los lados. - Las esquinas seguras se conservan, pero
son más pequeñas y difíciles de alcanzar.

Daño: 4 HP por impacto.

Implementación propuesta 1. Representar cada corriente como una
secuencia de bolas descendentes. 2. Calcular su posición en función del
tiempo para formar el patrón entrecruzado. 3. Definir huecos
transitables en el patrón. 4. Permitir que las bolas se acumulen abajo
si se desea reproducir ese detalle. 5. Implementar primero la versión
normal. 6. Añadir después el tablero más corto y las bolas laterales de
la variante difícil. 7. Reducir el tamaño de las zonas seguras en esa
variante.

Prioridad: alta. Es útil para entrenar navegación por rutas seguras.
2.3 Fire Helix --- hélice de fuego
Mecánica descrita - Corrientes de bolas de fuego más gruesas que las
de Fire Waves. - Son demasiado gruesas para atravesarlas. - No barren de
un lado a otro. - El SOUL puede mantenerse a salvo permaneciendo
cuidadosamente entre las corrientes.

Daño: 4 HP por impacto.

Implementación propuesta 1. Crear corrientes gruesas con posiciones
estables. 2. Dejar corredores seguros entre ellas. 3. Mantener las
corrientes sin barrido lateral. 4. Comprobar que los corredores tengan
espacio suficiente para el SOUL. 5. Entrenar al agente para localizar y
conservar una posición segura.

Prioridad: primera implementación de ataque. Es la más sencilla para
validar corrientes, colisiones y zonas seguras.
2.4 Faltering --- ataque fallido
Mecánica descrita - Bolas de fuego caen de forma irregular desde la
parte superior. - Si se acercan al SOUL, se alejan deliberadamente de
él. - Este ataque no puede causar daño. - Si el HP del protagonista es 2
o menos, Toriel utiliza exclusivamente este ataque. - Si un ataque
normal reduce el HP a 2, Toriel termina ese ataque inmediatamente. -
También usa Faltering después del undécimo Spare, independientemente de
la salud. - Después del duodécimo Spare deja de atacar.

Daño: ninguno.

Implementación propuesta 1. Generar bolas con posiciones y
velocidades iniciales variables. 2. Calcular la distancia entre cada
bola y el SOUL. 3. Si una bola se aproxima demasiado, cambiar su
dirección para alejarla. 4. Garantizar que no pueda colisionar con el
SOUL. 5. Añadir la selección automática de Faltering cuando HP ≤ 2. 6.
Interrumpir el ataque normal si el HP llega a 2. 7. Incorporar la regla
de los Spares en el controlador del combate.

Prioridad: implementar después de los ataques peligrosos, pero antes
de considerar completo el comportamiento del jefe.
3. Mecánicas adicionales del combate
HP y selección de ataques
- Mantener el HP del SOUL y aplicar el daño según el tipo de impacto.
- Con HP ≤ 2, Toriel pasa a Faltering.
- Si el HP vuelve a ser 3 o más, los ataques normales pueden regresar,
  salvo que ya se haya alcanzado la condición de Spares indicada.
- Si un ataque normal reduce el HP a 2, se interrumpe inmediatamente.
Spare
- La fuente indica que fallar los ataques tiene el mismo efecto que
  perdonar a Toriel.
- Después de 11 Spares, Toriel usa Faltering independientemente del
  HP.
- Después de 12 Spares deja de atacar.
Se recomienda incorporar el contador de Spare en una segunda etapa. No
es imprescindible para demostrar que el agente aprende a esquivar.
Fuera del alcance inicial
Huir y volver a iniciar el combate, diálogos, secuencias narrativas y
atacar a Toriel cuando deja de luchar no son necesarios para el objetivo
principal. Se pueden omitir inicialmente.
4. Arquitectura propuesta
toriel_rl/
├── main.py
├── environment.py
├── agent.py
├── attacks/
│   ├── __init__.py
│   ├── base_attack.py
│   ├── paw_sweep.py
│   ├── fire_waves.py
│   ├── fire_helix.py
│   └── faltering.py
├── entities/
│   ├── soul.py
│   ├── fireball.py
│   └── paw.py
├── training.py
├── visualization.py
├── evaluation.py
└── config.py
  Archivo                             Responsabilidad
  main.py                           Ejecutar una partida de prueba o
demostración.

  environment.py                    Estado del tablero, HP, turnos,
selección de ataque y reglas.

  agent.py                          Observación, selección de acciones
y aprendizaje.

  base_attack.py                    Interfaz común: iniciar,
actualizar, detectar colisiones y
terminar.

  paw_sweep.py                      Movimiento de patas, generación y
lanzamiento de bolas.

  fire_waves.py                     Corrientes entrecruzadas, huecos y
variante difícil.

  fire_helix.py                     Corrientes gruesas y corredores
seguros.

  faltering.py                      Bolas irregulares que se alejan del
SOUL.

  soul.py                           Posición, movimiento, límites y
tamaño del corazón.

  fireball.py                       Posición, velocidad, rebotes y
colisiones de proyectiles.

  paw.py                            Posición, trayectoria y colisión de
la pata.

  training.py                       Entrenamiento por episodios y
guardado de resultados.

  visualization.py                  Dibujo del tablero, SOUL, ataques y
HP con Pygame.

  evaluation.py                     Métricas y comparación de agentes.
config.py                         Dimensiones, velocidades, daños y
parámetros ajustables.
5. Diseño del entorno de aprendizaje por refuerzo
El entorno debe ofrecer tres operaciones:
- reset(): reinicia el episodio y devuelve la observación inicial.
- step(action): aplica una acción, avanza la simulación y devuelve
  observación, recompensa y si terminó el episodio.
- render(): dibuja el estado cuando la visualización está activada.
Observación (estado)
Para empezar, no entregar una imagen completa al agente. Usar variables
numéricas: - Posición horizontal y vertical del SOUL. - HP actual. -
Tipo y fase del ataque activo. - Posición y velocidad de los proyectiles
cercanos. - Distancia a los proyectiles más próximos.

Para Q-Learning tabular, discretizar la posición y resumir el peligro
por dirección (arriba, abajo, izquierda y derecha). Si se incluyen todos
los proyectiles individualmente, el espacio de estados puede crecer
demasiado.
Acciones iniciales
1. Quedarse quieto.
2. Moverse arriba.
3. Moverse abajo.
4. Moverse a la izquierda.
5. Moverse a la derecha.
Añadir diagonales solo cuando la versión inicial funcione.
Recompensas sugeridas
Estos valores son propuestas de diseño, no datos oficiales del juego.

  Situación                                Recompensa
  Paso sin recibir daño                          +0.1
Evitar un proyectil cercano                      +1
Recibir daño                                    -10
Morir                                           -20
Sobrevivir hasta el final del patrón             +5

La recompensa positiva por cada paso sin daño debe probarse con cuidado:
podría incentivar que el agente prolongue innecesariamente el episodio.
Comparar resultados con y sin ella.
6. Algoritmo recomendado
Primera etapa: Q-Learning
Adecuado si el tablero se discretiza en una cuadrícula, se resume la
información de proyectiles y se mantienen pocas acciones.