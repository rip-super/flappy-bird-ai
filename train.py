import neat, os, pickle, pygame, sys, time
from game import Bird, Pipe, Base, FLOOR, WINDOW_WIDTH, draw_window, WINDOW, gen

TICK_RATE = 60

def eval_genomes(genomes, config, pickle_file=None):
    global WINDOW, gen
    best_score = 0
    window = WINDOW
    gen += 1

    # Lists to hold the neural networks and birds
    nets = []
    birds = []
    ge = []

    # If a pickle file is provided, load the neural network from it and use one bird
    if pickle_file and os.path.exists(pickle_file):
        print(f"Loading neural network from {pickle_file}...")
        with open(pickle_file, "rb") as f:
            best_net = pickle.load(f)
        nets.append(best_net)
        birds.append(Bird(230, 350))  # Use one bird when loading from the pickle file
    else:
        # Proceed with normal genome initialization (use all genomes)
        for genome_id, genome in genomes:
            genome.fitness = 0  # Start with fitness level of 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            birds.append(Bird(230, 350))
            ge.append(genome)

    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0

    clock = pygame.time.Clock()

    run = True
    while run and len(birds) > 0:
        clock.tick(TICK_RATE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

        for x, bird in enumerate(birds):
            if not pickle_file:  # Only update fitness if not using pre-trained network
                ge[x].fitness += 0.01
            bird.move()

            # Send bird's current state to the neural network and get the decision (jump or not)
            output = nets[x].activate(
                (bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom))
            )

            if output[0] > 0.5:  # Neural network output threshold for jumping
                bird.jump()

        base.move()

        remove = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            for bird in birds[:]:  # Iterate over a copy of the birds list
                if pipe.collide(bird, window):
                    bird_index = birds.index(bird)  # Get the index before removing
                    if not pickle_file:  # Only modify fitness if not using pre-trained network
                        ge[bird_index].fitness -= 1
                    birds.pop(bird_index)  # Remove the bird
                    nets.pop(bird_index)   # Remove the corresponding network
                    if not pickle_file:
                        ge.pop(bird_index)   # Remove the genome

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                remove.append(pipe)

            if not pipe.passed and len(birds) > 0 and pipe.x < birds[0].x:  # Make sure there's at least 1 bird
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            if not pickle_file:  # Only update fitness if not using pre-trained network
                for genome in ge:
                    genome.fitness += 1
            pipes.append(Pipe(WINDOW_WIDTH))
        
        if score > best_score:
            best_score = score

        for r in remove:
            pipes.remove(r)

        for bird in birds[:]:  # Iterate over a copy of the birds list
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                bird_index = birds.index(bird)  # Get the index before removing
                birds.pop(bird_index)  # Remove the bird
                nets.pop(bird_index)   # Remove the corresponding network
                if not pickle_file:
                    ge.pop(bird_index)  # Remove the genome

        draw_window(window, birds, pipes, base, score, gen, pipe_ind, best_score)

        # Save the best neural network if score exceeds threshold and terminate the program
        if score == 50 and not pickle_file:
            time.sleep(0.5)
            print("Saving best neural network and terminating the program...")
            if not pickle_file:  # Only save if not using a pre-loaded network
                pickle.dump(nets[0], open("best.pickle", "wb"))
            run = False
            sys.exit()

def run(config_file, pickle_file=None):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # If a .pickle file is provided, don't run the full NEAT algorithm.
    if pickle_file and os.path.exists(pickle_file):
        # Simulate using the pre-trained network by calling eval_genomes with the pickle file
        eval_genomes([], config, pickle_file=pickle_file)
    else:
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        # Run the NEAT algorithm to evolve the population for up to 50 generations.
        winner = p.run(eval_genomes, 50)

        # Show final stats
        print('\nBest genome:\n{!s}'.format(winner))