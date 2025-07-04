def phase_freeze_training(current_epoch, phase_counter):
    epochs_sequence = [10, 21, 55]

    phase_lengths_sequence = [3, 7]
    assert len(epochs_sequence) == len(phase_lengths_sequence), \
        "Length of epochs_sequence must be equal to phase_lengths_sequence. Please change the configurations."
            
    phase_param_groups = ["A", "B", "C"]
    num_param_groups = len(phase_param_groups)

    epoch_accumulator = 0
    for (epochs, phase_length) in zip(epochs_sequence, phase_lengths_sequence):
        # Determine the length of the current phase of epochs (sequence lr scheduler)
        if current_epoch < epoch_accumulator + epochs:
            # How far into the current LR phase we are
            local_epoch = current_epoch - epoch_accumulator
            
            if current_epoch == epoch_accumulator:
                phase_counter = 0
                
            # Chek if we enter a new phase which leads to unfreezing respective parameters
            if local_epoch % phase_length == 0:
                # Calculate index to determine which group of parameters will be unfrozen
                phase_group_index = phase_counter % num_param_groups 
                print("Unfreeze group: ", phase_param_groups[phase_group_index])
                phase_counter += 1
                   
            return phase_counter
        
        # Move to next epoch sequence in lr scheduler
        epoch_accumulator += epochs


phase_counter = 0                       
epoch = 0

while epoch < sum([10, 21, 55]):
    print("Epoch: ", epoch, phase_counter)
    phase_counter=phase_freeze_training(epoch, phase_counter)
    epoch += 1

