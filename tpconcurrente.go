package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"
)

// Estructura para representar cada fila del CSV
type Atencion struct {
	Mes                   int    // Mes de la atención
	Dia                   int    // Día de la atención
	NombreEstablecimiento string // Nombre del establecimiento de salud
	Atendidos             int    // Número de pacientes atendidos
	Atenciones            int    // Número total de atenciones
}

// Nodo del árbol de decisión
type Node struct {
	Feature    string // Característica en la que se basará la división (e.g., Mes, Dia)
	Threshold  int    // Umbral de división para la característica
	Left       *Node  // Rama izquierda (datos que cumplen la condición)
	Right      *Node  // Rama derecha (datos que no cumplen la condición)
	IsLeaf     bool   // Indica si es un nodo hoja
	Prediction bool   // Predicción para este nodo (true = congestionado, false = no congestionado)
}

// Estructura del árbol de decisión
type DecisionTree struct {
	Root *Node // Nodo raíz del árbol
}

// Constructor para un nuevo árbol de decisión
func NewDecisionTree() *DecisionTree {
	return &DecisionTree{Root: &Node{}} // Inicializa un nuevo árbol con un nodo raíz vacío
}

// Función para entrenar un árbol de decisión con datos
func (dt *DecisionTree) Train(data []Atencion) {
	dt.Root = dt.buildTree(data, 0) // Comienza a construir el árbol desde la raíz
}

// Función recursiva para construir el árbol
func (dt *DecisionTree) buildTree(data []Atencion, depth int) *Node {
	if len(data) < 10 || depth > 5 { // Condición de parada: si hay pocos datos o se alcanzó la profundidad máxima
		return &Node{
			IsLeaf:     true,                    // Este es un nodo hoja
			Prediction: dt.makePrediction(data), // Se hace una predicción basada en los datos
		}
	}

	// Selección aleatoria de la característica y umbral
	feature, threshold := dt.selectFeatureAndThreshold()
	leftData, rightData := dt.splitData(data, feature, threshold) // Dividir los datos en dos grupos

	// Crear un nuevo nodo con la característica y umbral seleccionados
	node := &Node{
		Feature:   feature,
		Threshold: threshold,
	}
	node.Left = dt.buildTree(leftData, depth+1)   // Construir rama izquierda
	node.Right = dt.buildTree(rightData, depth+1) // Construir rama derecha

	return node // Retornar el nodo construido
}

// Función para seleccionar una característica y umbral aleatorio
func (dt *DecisionTree) selectFeatureAndThreshold() (string, int) {
	features := []string{"Mes", "Dia", "Atendidos", "Atenciones"} // Características posibles
	feature := features[rand.Intn(len(features))]                 // Selección aleatoria de una característica
	threshold := rand.Intn(12) + 1                                // Generar un umbral aleatorio entre 1 y 12
	return feature, threshold
}

// Función para dividir los datos basados en la característica y umbral
func (dt *DecisionTree) splitData(data []Atencion, feature string, threshold int) ([]Atencion, []Atencion) {
	var left, right []Atencion // Inicializar slices para los datos divididos
	for _, att := range data {
		switch feature {
		case "Mes":
			if att.Mes <= threshold { // Comparar con el umbral
				left = append(left, att) // Agregar a la rama izquierda
			} else {
				right = append(right, att) // Agregar a la rama derecha
			}
		case "Dia":
			if att.Dia <= threshold {
				left = append(left, att)
			} else {
				right = append(right, att)
			}
		case "Atendidos":
			if att.Atendidos <= threshold {
				left = append(left, att)
			} else {
				right = append(right, att)
			}
		case "Atenciones":
			if att.Atenciones <= threshold {
				left = append(left, att)
			} else {
				right = append(right, att)
			}
		}
	}
	return left, right // Retornar los datos divididos
}

// Hacer una predicción basada en los datos
func (dt *DecisionTree) makePrediction(data []Atencion) bool {
	if len(data) == 0 {
		// Si no hay datos, devolvemos false o alguna predicción por defecto
		return false
	}

	total := 0
	for _, att := range data {
		total += att.Atendidos // Sumar el total de atendidos
	}
	avg := total / len(data) // Calcular el promedio

	// Considerar congestión si el promedio de "Atendidos" es mayor a 20
	return avg > 20
}

// Predicción del árbol para un nuevo conjunto de datos
func (dt *DecisionTree) Predict(att Atencion) bool {
	node := dt.Root    // Comenzar desde la raíz
	for !node.IsLeaf { // Mientras no sea un nodo hoja
		switch node.Feature {
		case "Mes":
			if att.Mes <= node.Threshold {
				node = node.Left // Seguir por la rama izquierda
			} else {
				node = node.Right // Seguir por la rama derecha
			}
		case "Dia":
			if att.Dia <= node.Threshold {
				node = node.Left
			} else {
				node = node.Right
			}
		case "Atendidos":
			if att.Atendidos <= node.Threshold {
				node = node.Left
			} else {
				node = node.Right
			}
		case "Atenciones":
			if att.Atenciones <= node.Threshold {
				node = node.Left
			} else {
				node = node.Right
			}
		}
	}
	return node.Prediction // Retornar la predicción del nodo hoja
}

// Estructura del bosque aleatorio
type RandomForest struct {
	Trees []*DecisionTree // Slice que contiene los árboles de decisión
	mu    sync.Mutex      // Mutex para sincronización de acceso concurrente
}

// Función para entrenar un bosque aleatorio
func (rf *RandomForest) Train(data []Atencion) {
	var wg sync.WaitGroup
	rf.Trees = make([]*DecisionTree, 0, numTrees)     // Inicializamos el slice de árboles con capacidad para numTrees
	treeChannel := make(chan *DecisionTree, numTrees) // Canal para enviar los árboles entrenados

	// Entrenar los árboles en paralelo
	for i := 0; i < numTrees; i++ {
		wg.Add(1) // Aumentar el contador de goroutines
		go func() {
			defer wg.Done() // Decrementar el contador al finalizar

			subData := sampleData(data) // Obtener una muestra de datos
			tree := NewDecisionTree()   // Crear un nuevo árbol
			tree.Train(subData)         // Entrenar el árbol con los datos muestreados
			treeChannel <- tree         // Enviar el árbol entrenado al canal
		}()
	}

	// Recolectar los árboles entrenados
	go func() {
		wg.Wait()          // Esperar a que todas las goroutines terminen
		close(treeChannel) // Cerrar el canal
	}()

	for tree := range treeChannel {
		rf.mu.Lock()                      // Bloquear el acceso al slice de árboles
		rf.Trees = append(rf.Trees, tree) // Agregar el árbol entrenado al slice
		rf.mu.Unlock()                    // Desbloquear el acceso
	}
}

// Función que toma una muestra aleatoria de los datos
func sampleData(data []Atencion) []Atencion {
	trainSize := int(float64(len(data)) * 0.8) // Calcular el tamaño de la muestra (80% de los datos)
	rand.Shuffle(len(data), func(i, j int) {   // Mezclar los datos
		data[i], data[j] = data[j], data[i]
	})
	return data[:trainSize] // Retornar la muestra
}

// Predicción del bosque aleatorio
func (rf *RandomForest) Predict(establishment string, month int, day int) bool {
	if len(rf.Trees) == 0 { // Verificar si hay árboles entrenados
		return false
	}

	votes := 0 // Contador de votos a favor de congestión
	for _, tree := range rf.Trees {
		// Crear una nueva instancia de Atencion para la predicción
		testAtencion := Atencion{
			Mes:                   month,
			Dia:                   day,
			NombreEstablecimiento: establishment,
		}

		// Hacer la predicción con el árbol actual
		if tree.Predict(testAtencion) {
			votes++ // Incrementar el conteo de votos si se predice congestión
		}
	}

	// Retornar true si la mayoría de los árboles predicen congestión
	return votes > len(rf.Trees)/2
}

// Número de árboles para el bosque aleatorio
var numTrees int          // Se definirá según la entrada del usuario
var atenciones []Atencion // Lista global de atenciones procesadas

// Función principal
func main() {
	rf := &RandomForest{} // Crear una nueva instancia del bosque aleatorio

	for {
		// Mostrar el menú de opciones al usuario
		fmt.Println("\nMenú:")
		fmt.Println("1. Procesar registros")
		fmt.Println("2. Entrenar algoritmo")
		fmt.Println("3. Predecir congestión en un establecimiento")
		fmt.Println("4. Salir")
		fmt.Print("Escoge tu opción: ")

		var option int
		fmt.Scan(&option) // Leer la opción del usuario

		// Evaluar la opción seleccionada
		switch option {
		case 1:
			// Procesar registros solo si no se han procesado previamente
			if len(atenciones) == 0 {
				fmt.Println("Procesando registros...")
				start := time.Now() // Iniciar el temporizador para medir el tiempo de procesamiento

				// Abrir el archivo CSV que contiene los registros
				file, err := os.Open("atenciones_filtradas.csv")
				if err != nil {
					log.Fatal(err) // Manejar error si no se puede abrir el archivo
				}
				defer file.Close() // Asegurarse de cerrar el archivo al final

				reader := csv.NewReader(file) // Crear un lector CSV
				reader.Comma = ','            // Establecer el separador de columnas

				// Leer y verificar la cabecera del CSV
				if _, err := reader.Read(); err != nil {
					log.Fatalf("Error al leer la cabecera: %v", err)
				}

				var wg sync.WaitGroup                   // Grupo de espera para sincronizar goroutines
				dataChannel := make(chan Atencion, 100) // Canal para enviar datos de atención procesados

				// Goroutine para leer registros del CSV y procesarlos
				go func() {
					for {
						record, err := reader.Read() // Leer cada registro del archivo
						if err != nil {
							break // Salir si no hay más registros
						}

						// Verificar que el registro tiene al menos 5 columnas
						if len(record) < 5 {
							fmt.Println("Fila inválida: ", record) // Mostrar mensaje de error para fila inválida
							continue                               // Saltar a la siguiente iteración
						}

						wg.Add(1) // Aumentar el contador de goroutines
						go func(record []string) {
							defer wg.Done() // Decrementar el contador al finalizar

							// Convertir los valores del registro a tipos adecuados
							mes, err := strconv.Atoi(record[0])
							if err != nil {
								log.Printf("Error al convertir mes: %v", err)
								return
							}
							dia, err := strconv.Atoi(record[1])
							if err != nil {
								log.Printf("Error al convertir dia: %v", err)
								return
							}
							atendidos, err := strconv.Atoi(record[3])
							if err != nil {
								log.Printf("Error al número de atendidos: %v", err)
								return
							}
							atencionesCount, err := strconv.Atoi(record[4])
							if err != nil {
								log.Printf("Error al número de atenciones: %v", err)
								return
							}

							// Crear un nuevo objeto Atencion con los datos procesados
							data := Atencion{
								Mes:                   mes,
								Dia:                   dia,
								NombreEstablecimiento: record[2],
								Atendidos:             atendidos,
								Atenciones:            atencionesCount,
							}
							dataChannel <- data // Enviar el objeto Atencion al canal
						}(record)
					}
					wg.Wait()          // Esperar a que todas las goroutines terminen
					close(dataChannel) // Cerrar el canal
				}()

				// Recibir los datos del canal y agregarlos al slice atenciones
				for data := range dataChannel {
					atenciones = append(atenciones, data) // Agregar datos procesados al slice
				}

				// Mostrar información sobre el procesamiento
				fmt.Printf("Registros procesados: %d\n", len(atenciones))
				duration := time.Since(start) // Calcular el tiempo de procesamiento
				fmt.Printf("Tiempo de procesamiento: %v\n", duration)
			} else {
				// Mensaje si los registros ya fueron procesados
				fmt.Println("Los registros ya han sido procesados.")
			}

		case 2:
			// Entrenar el algoritmo solo si se han procesado los registros
			if len(atenciones) == 0 {
				fmt.Println("Primero debes procesar los registros.") // Mensaje de advertencia
			} else {
				// Solicitar al usuario el número de árboles para entrenar el algoritmo
				fmt.Print("Ingresa el número de árboles para entrenar el algoritmo: ")
				fmt.Scan(&numTrees)

				start := time.Now()           // Iniciar el temporizador para el entrenamiento
				rf.Train(atenciones)          // Entrenar el bosque aleatorio con los registros procesados
				duration := time.Since(start) // Calcular el tiempo de entrenamiento
				fmt.Printf("Algoritmo entrenado con %d árboles en %v\n", numTrees, duration)
			}
		case 3:
			if len(rf.Trees) == 0 {
				fmt.Println("Primero debes entrenar el algoritmo.")
			} else {
				// Mapa para almacenar los establecimientos únicos y un slice para mantener el orden
				uniqueEstablishments := make(map[string]struct{})
				var establishmentsList []string // Slice para mantener la lista de establecimientos en orden

				// Recorremos las atenciones y llenamos el mapa y el slice
				for _, att := range atenciones {
					// Verificamos si el establecimiento ya está en el mapa
					if _, exists := uniqueEstablishments[att.NombreEstablecimiento]; !exists {
						uniqueEstablishments[att.NombreEstablecimiento] = struct{}{}               // Marcamos el establecimiento como existente
						establishmentsList = append(establishmentsList, att.NombreEstablecimiento) // Agregamos al slice
					}
				}

				// Imprimimos la lista de establecimientos disponibles
				fmt.Println("Establecimientos disponibles:")
				for i, establishment := range establishmentsList {
					fmt.Printf("%d. %s\n", i+1, establishment) // Mostramos el índice y el nombre del establecimiento
				}

				// Pedimos al usuario que seleccione un establecimiento
				fmt.Print("Selecciona el número del establecimiento: ")
				var index int
				fmt.Scan(&index) // Leemos la opción del usuario

				// Validamos si el índice está en el rango de la lista
				if index < 1 || index > len(establishmentsList) {
					fmt.Println("Número inválido.") // Mensaje de error si el número no es válido
					break
				}

				// Seleccionamos el establecimiento de acuerdo al índice ingresado
				selectedEstablishment := establishmentsList[index-1] // Obtenemos el establecimiento por índice

				// Pedimos al usuario que ingrese el mes y el día para la predicción
				fmt.Print("Ingresa el mes (1-12): ")
				var month int
				fmt.Scan(&month) // Leemos el mes
				fmt.Print("Ingresa el día (1-31): ")
				var day int
				fmt.Scan(&day) // Leemos el día

				// Realizamos la predicción usando el bosque aleatorio
				if rf.Predict(selectedEstablishment, month, day) {
					// Mostramos el resultado de la predicción
					fmt.Printf("El establecimiento %s estará congestionado.\n", selectedEstablishment)
				} else {
					// Mostramos el resultado de la predicción
					fmt.Printf("El establecimiento %s no estará congestionado.\n", selectedEstablishment)
				}
			}
		case 4:
			// Mensaje de despedida y salir del programa
			fmt.Println("Saliendo...")
			return
		default:
			// Mensaje de error si la opción no es válida
			fmt.Println("Opción no válida, intenta de nuevo.")
		}
	}
}
