import math
import pandas as pd

def distanceCalculator(A,B):
    distance =  math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    return distance
def degreeCalc(A,B,C):
   # Calcolo dei vettori BA e BC
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    prodScalare= BA[0] * BC[0] + BA[1] * BC[1]
    # Norme dei vettori
    normaBA = math.sqrt(BA[0] ** 2 + BA[1] ** 2)
    normaBC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)
    if normaBA == 0 or normaBC == 0:
        return 0.0
    cos = prodScalare / (normaBA * normaBC)
    cos = max(-1, min(1,cos))
    # Calcolo dell'angolo in gradi
    angleRad = math.acos(cos)
    angleDeg = math.degrees(angleRad)
    return angleDeg

def checkSmash(A,B,C):
    distanceAB=distanceCalculator(A,B)
    distanceBC=distanceCalculator(B,C)
    if distanceAB == 0:  # Evitiamo divisioni per zero
        percentChange = 0
    else:
        percentChange = ((distanceBC - distanceAB) / distanceAB) * 100
    # Parametri per rilevare uno smash
    limitChange = 40 # La variazione percentuale minima per considerare uno smash
    limitXVariation = 30 #limite per la variazione orizzontale
    minDistance=40 # variazione fissa
    if percentChange > limitChange:
        # Inoltre, verifica se la variazione in X è piccola
        deltaXAB = abs(B[0] - A[0])
        deltaXBC = abs(C[0] - B[0])
        # Se la distanza aumenta significativamente e X rimane stabile, allora è uno smash
        if (distanceAB>=minDistance or distanceBC>=minDistance) and deltaXAB < limitXVariation and deltaXBC < limitXVariation:
            if C[1] > B[1]:  # La pallina deve muoversi verso il basso
                return True
    return False
def checkBounce(A,B,C,D):
    if(distanceCalculator(B,C)<8):
        B = ((B[0] + C[0]) / 2, (B[1] + C[1]) / 2)
        C = D
        degrees=degreeCalc(A,B,C)
    degrees=degreeCalc(A,B,C)
    limitDegrees=40
    limitYVariation=8
    #se la pallina precedente o successiva è fuori dallo schermo, non faccio nessun conto
    if(A[1]==0 or C[1]==0):
        return 0
    #indetificare quando la pallina cambia direzione significativamente
    elif limitDegrees<degrees<180-limitDegrees:
        if((distanceCalculator(A,B)+distanceCalculator(B,C)>15)and distanceCalculator(A,B)>3 and distanceCalculator(B,C)>3):
            return 2 #Marca il frame come variazione di angolo
    # Identificare un cambio di direzione in Y (la pallina cambia velocità in verticale)
    elif (A[1] < B[1] > C[1]) or (A[1] > B[1] < C[1]) and (abs(B[1] - A[1]) >= limitYVariation or abs(B[1] - C[1]) >= limitYVariation):
        return 1
    elif checkSmash(A,B,C):
        return 3
    return 0


def bouncerDetector(predicted_csv):
    # Converti l'input in un DataFrame
    predicted_csv = pd.DataFrame(predicted_csv)

    # Estraggo 2 array X e Y
    predicted_X = predicted_csv['X']
    predicted_Y = predicted_csv['Y']
    
    # Scorri i dati e rileva i cambiamenti di direzione
    for i in range(1, len(predicted_csv) - 2):
        A = (predicted_X[i-1], predicted_Y[i-1])  # Punto precedente
        B = (predicted_X[i], predicted_Y[i])      # Punto corrente
        C = (predicted_X[i+1], predicted_Y[i+1])  # Punto successivo
        D = (predicted_X[i+2], predicted_Y[i+2])  # Punto successivo

        
        # Applica il controllo dei rimbalzi
        predicted_csv.loc[i,'Bounce'] = checkBounce(A, B, C, D)
        # Evita doppi rimbalzi consecutivi
        if predicted_csv.loc[i-1,'Bounce']> 0:
            predicted_csv.loc[i,'Bounce'] = 0
    return predicted_csv