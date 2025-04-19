# the plan 
# for n iterations do (every iteration is a t instant)
    # for each ant do 
        # select a starting task "at random /following a certain heuristic"
        # for every none chosen task yet
            # pick a task following the "regle aleatoire de transition proportionelle"
        # calculate delta sigma so that you add it later
    # pheromone evaporation + pheromone adding from all the ants


# how to model the flow shop problem on a graph ?
# every ant takes the first task at random...
# an ant that picked one task cannot repick it again (tabou list logic)
# at any instant t, an ant can pick the next job taking into account these two rules :
#   rule 1 : heuristic related to the flow shop problem, to define later
#   rule 2 : the pheromone quantity on the edge from the current job to the next job
#   two parameters alpha (for pheromones) and beta (for projected problem heuristic) decide how important is each of the two rules
# once the job list complete, the ant calculates the completion time "already defined function"
# then according to this value, the ant updates the pheromone on all the edges that together create the job sequence
# the ant will do many of these spins, after each spin, the ant will update its memory with the best job sequence it found so far


# given n possible jobs to pick next, the ant follows this stochatstic decision rule:
# probabilty(nextJob=j | current = i) = ( sigmaij(t)^alpha * etaij^beta ) /  SUM (what is in the numerator for each possible next job j)
# sigmaij(t) = pheromone value from i to j at the iteration t
# etaij also called visiblity = a value that is related to the flow shop problem, the bigger the better, example : if we are prioritizing short jobs first, eta would be the inverse of the completion time of that specefic job on a specific machine
# what are the value ranges for alpha and beta ? 
# small alpha means that the algorithm follows visibility most of the time favoring a greedy algorithm
# small beta means that ants will follow high pheromone intensity paths implying fast convergence ( less exploration and only exploitation)
# at the end of a completed sequence, each ant calculates delta sigma ij, which is the ammount of phermones to add to sigma ij
# every delta sigma ij that the ant passed through = Q / completion time of the entire job sequence
# after all the ants have generated one sequence, its time for pheromone update :
# sigma ij (t+1)= (1-ρ)* sigmaij(t) + all delta sigma ij from the different ants m
# ρ = 0 means no evaporation at all, sigma just keeps adding up and we will have early convergence
# ρ = 1 means that there no learning going on since any pheromone trace from the previous iteration is wiped out




# parameters to tune so far : 
# alpha : how important is the pheromon effect
# beta : how important is the visibility effect
# the visibility formula
# Q : update intensity "bigger Q means bigger update values to sigma ij"
# ρ : evaporation rate : [0,1]
# m : number of ants
# sigma 0 : initial pheromone intensity


# that was variant 0
# variant 1 also called elitisme, that allows ants that found the best path to deposit more pheromone "their Q is bigger or something"
# variant 2 (min-max ant system), only the ant that found the best path in that iteration is allowed to deposit its pheromones, also the value of pheromones is kind of capped from top and bottom
# all of them start at max value, then when updated, edges with smaller values take more uplift effect than the edges with bigger values
# variant 3 Ant colony system, candidate list logic (reduce the amount of jobs to pick from at each step) + when every ant is updating their best path, every edge that is part of the best path among all ants gets a bonus



# the specefic variation we are coding is ANT SYSTEM (AS) "defined in the plan at the top"
# the elitist AS is an easy variation to implement as well, we will add that
# ACS is a bit more complex to implement, but i'll try to do it anyways
# welp looks like ACS is more sphisticated than I thought ... gotta give it a deeper read



