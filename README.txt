PROJET GO 8x8 - RENDU TOURNOI
=============================

Binome
------
Aymane Kechna,Pierre-Louis Shrestha--Salie



Techniques utilisees dans l'IA (`myPlayer.py`)
----------------------------------------------
1. Recherche Negamax alpha-beta avec approfondissement iteratif.
   - On lance des recherches de profondeur 1, puis 2, puis 3, etc.
   - A chaque profondeur completee, on garde un coup valide.
   - Cela permet de toujours jouer un coup meme si le temps est coupe.
2. PVS (Principal Variation Search / NegaScout) pour accelerer l'exploration.
   - Le premier coup d'un noeud est evalue avec une fenetre alpha-beta complete.
   - Les coups suivants sont testes avec une fenetre plus etroite (null-window),
     puis re-analyses seulement si necessaire.
   - Resultat: meme qualite de choix, avec moins de noeuds explores.
3. Table de transposition avec bornes EXACT / LOWER / UPPER.
   - Les positions deja vues (hash Zobrist) sont memorisees.
   - EXACT = valeur precise, LOWER/UPPER = bornes apres coupure.
   - Cela evite de recalculer des sous-arbres identiques.
4. Fenetres d'aspiration a partir du score de la profondeur precedente.
   - On recherche d'abord autour du score precedent avec une petite marge.
   - Si la vraie valeur sort de cette marge, on relance en fenetre complete.
   - Cela accelere la recherche dans les cas stables.
5. Ordonnancement des coups avec heuristique tactique locale.
   - Priorite aux coups tactiques: capture, mise en atari, sauvetage de groupe.
   - Les coups faibles (ex: PASS trop tot) sont explores plus tard.
   - Un bon ordre de coups augmente fortement les coupures alpha-beta.
6. Killer moves + history heuristic pour ameliorer les coupures beta.
   - Killer moves: coups ayant deja provoque une coupure a la meme profondeur.
   - History heuristic: score cumulatif des coups souvent efficaces.
   - Ces informations servent a reclasser les coups avant exploration.
7. Evaluation heuristique du plateau combinant:
   - score de territoire (`compute_score`),
   - captures,
   - libertes de groupes (union-find),
   - groupes en atari / proche atari,
   - influence locale,
   - controle du centre,
   - detection d'yeux.
   - L'evaluation donne une estimation rapide de la qualite d'une position
     sans attendre la fin de partie.
8. Gestion dynamique du temps par coup selon le temps total restant.
   - Le budget par coup depend du temps deja consomme et du nombre de cases vides.
   - L'IA investit plus de temps quand la position devient tactiquement critique
     (milieu/fin de partie), tout en gardant une marge de securite.


Execution
---------
- Match myPlayer vs myPlayer:
  `python3 localGame.py`

- Match myPlayer vs random:
  `python3 namedGame.py myPlayer randomPlayer`
