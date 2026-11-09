import numpy as np

def calculer_champs_panofsky_phillips(rho, J, dJ_dt, n, r):
    """
    Calcule la contribution élémentaire aux champs électrique E et magnétique B 
    générée par un élément de volume source dV' selon les équations de Panofsky-Phillips.
    
    Toutes les sources (rho, J, dJ_dt) doivent être évaluées 
    à l'instant retardé t' = t - r/c propre à cet élément source.
    """
    epsilon_0 = 8.8541878128e-12  # Permittivité du vide [F/m]
    c = 299792458.0               # Vitesse de la lumière [m/s]
    
    if r == 0:
        return np.zeros(3), np.zeros(3) # Évite la divergence à l'origine locale
        
    n = n / np.linalg.norm(n)
    prefacteur = 1.0 / (4.0 * np.pi * epsilon_0)

    # 1. CHAMP ÉLECTRIQUE ÉLÉMENTAIRE (dE)
    terme_position = rho * n
    terme_vitesse = (np.dot(J, n) * n + np.cross(np.cross(J, n), n)) / c
    dE_proche = prefacteur * (terme_position + terme_vitesse) / (r**2)
    
    terme_acceleration = np.cross(np.cross(dJ_dt, n), n) / (c**2)
    dE_lointain = prefacteur * terme_acceleration / r
    
    dE = dE_proche + dE_lointain

    # 2. CHAMP MAGNÉTIQUE ÉLÉMENTAIRE (dB)
    dB_proche = prefacteur * np.cross(J, n) / ((c**2) * (r**2))
    dB_lointain = prefacteur * np.cross(dJ_dt, n) / ((c**3) * r)
    
    dB = dB_proche + dB_lointain

    return dE, dB

def integrer_champs_panofsky_phillips(points_sources, volumes_dV, rho_global, J_global, dJ_dt_global, point_M):
    """
    Réalise l'INTEGRALE VOLUMIQUE numérique des équations de Panofsky-Phillips
    au point d'observation M en sommant les contributions de toutes les sources.
    
    Paramètres :
    -----------
    points_sources : np.ndarray -> Matrice (N, 3) des coordonnées [x, y, z] de chaque élément source
    volumes_dV     : np.ndarray -> Tableau (N,) contenant la valeur du volume dV' de chaque élément [m^3]
    rho_global     : np.ndarray -> Tableau (N,) des densités de charge au temps retardé t'
    J_global       : np.ndarray -> Matrice (N, 3) des densités de courant au temps retardé t'
    dJ_dt_global   : np.ndarray -> Matrice (N, 3) des dérivées temporelles du courant au temps retardé t'
    point_M        : np.ndarray -> Coordonnées [x, y, z] du point d'observation M [m]
    """
    E_total = np.zeros(3)
    B_total = np.zeros(3)
    
    # Boucle d'intégration numérique sur l'ensemble du volume source
    for i in range(len(points_sources)):
        # Calcul du vecteur reliant l'élément source i au point M
        r_vec = point_M - points_sources[i]
        r = np.linalg.norm(r_vec)
        
        if r > 0:
            n = r_vec / r
            # Calcul des contributions élémentaires dE et dB
            dE, dB = calculer_champs_panofsky_phillips(
                rho_global[i], J_global[i], dJ_dt_global[i], n, r
            )
            # Sommation (Intégration) pondérée par le volume dV'
            E_total += dE * volumes_dV[i]
            B_total += dB * volumes_dV[i]
            
    return E_total, B_total

# =========================================================================
# Validation de la simulation avec intégrale volumique numérique
# =========================================================================
if __name__ == "__main__":
    print("--- Validation de l'Intégrale Volumique de Panofsky-Phillips ---")
    
    # Création d'une petite distribution de volume source (grille 2x2x2)
    x, y, z = np.meshgrid([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5])
    sources = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
    N_elements = len(sources)
    
    # Chaque élément a un volume élémentaire dV' = 1 m^3
    dV = np.ones(N_elements)
    
    # Distribution des sources homogènes au temps retardé
    rho_distribution = np.ones(N_elements) * 1.5e-9
    J_distribution = np.zeros((N_elements, 3))
    J_distribution[:, 0] = 2.0  # Courant constant selon X
    dJ_dt_distribution = np.zeros((N_elements, 3))
    dJ_dt_distribution[:, 0] = 5.0e7  # Accélération forte selon X
    
    # Point d'observation M éloigné sur l'axe Z (zone de rayonnement)
    M = np.array([0.0, 0.0, 300.0])
    
    # Calcul des champs finaux par intégration
    E_final, B_final = integrer_champs_panofsky_phillips(
        sources, dV, rho_distribution, J_distribution, dJ_dt_distribution, M
    )
    
    print(f"Intégrale du champ électrique E total au point M : {E_final} V/m")
    print(f"Intégrale du champ magnétique B total au point M : {B_final} T")
