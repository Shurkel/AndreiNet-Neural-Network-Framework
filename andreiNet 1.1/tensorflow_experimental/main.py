import tensorflow as tf
import numpy as np
import os
import time
import requests
import argparse

# --- Parse command line arguments ---
parser = argparse.ArgumentParser(description='Text Generation with TensorFlow')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'], 
                    help='Mode to run: "train" or "generate"')
parser.add_argument('--checkpoint', type=str, default=None, 
                    help='Path to specific checkpoint to load (default: use latest)')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train (default: 10)')
parser.add_argument('--seed', type=str, default="ROMEO:",
                    help='Seed text for generation (default: "ROMEO:")')
parser.add_argument('--length', type=int, default=500,
                    help='Length of text to generate (default: 500)')
parser.add_argument('--temp', type=float, default=0.8,
                    help='Temperature for generation (default: 0.8)')

args = parser.parse_args()

print("TensorFlow Version:", tf.__version__)

# --- Check for GPU ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"\n{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs detected and configured.\n")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("\nNo GPU detected. Running on CPU (will be slower).\n")


# --- 1. Prepare Data ---
url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
response = requests.get(url)
response.raise_for_status()
#text demo written by hand ignore Shakespeare on multiple lines
'''
text = """H
Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.

I am peeing my pants. Since I am a Capulet,
I am not allowed to love you.
But I love you, Romeo.
I love you so much that I am willing to die for you.
Cheerios!
I am a Capulet, and I am not allowed to love you.
I spilled my Cheerios all over the floor.

"""
'''
text=response.text
# Print text size for debugging
print(f"Text length: {len(text)} characters")


vocab = sorted(list(set(text)))
vocab_size = len(vocab)
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# --- 2. Create Training Sequences ---

# Adjust sequence length based on text size
# For small texts, use a much smaller sequence length
text_length = len(text)
if text_length < 1000:
    seq_length = 10  # Use a small sequence length for tiny texts
    # Repeat the text to get more training data
    repetitions = max(1, 1000 // text_length)
    text_as_int = np.tile(text_as_int, repetitions)
    print(f"Text repeated {repetitions} times to get more training data")
    print(f"New text length: {len(text_as_int)} characters")
elif text_length < 10000:
    seq_length = 25  # Medium sequence length for medium texts
else:
    seq_length = 50  # Larger sequence length for large texts

examples_per_epoch = len(text_as_int) // (seq_length + 1)
print(f"Sequence length: {seq_length}, Examples per epoch: {examples_per_epoch}")

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Adjust batch size based on dataset size
# Calculate a safe batch size that won't result in empty batches
num_sequences = examples_per_epoch
print(f"Number of sequences: {num_sequences}")

if num_sequences < 128:
    BATCH_SIZE = max(1, num_sequences // 2)  # Use half the sequences, minimum 1
else:
    BATCH_SIZE = 128

print(f"Using batch size: {BATCH_SIZE}")

BUFFER_SIZE = min(10000, num_sequences * 2)  # Adjust buffer size

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=False)  # Changed to False to avoid empty batches
    .prefetch(tf.data.experimental.AUTOTUNE))

# Check if dataset is empty
is_empty = True
for _ in dataset.take(1):
    is_empty = False
    break

if is_empty:
    raise ValueError("Dataset is empty. Try using more text or smaller batch size.")

# --- 3. Build Model Function ---
embedding_dim = 64   # WAS 256
rnn_units = 256     # WAS 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        # Option: Replace LSTM with GRU for potential minor speedup
        # tf.keras.layers.GRU(rnn_units,
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=False,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

# --- 4. Model Loading and Saving Functions ---
def save_model(model, save_dir="./saved_models/text_gen_model"):
    """Save the entire model (weights + architecture) for later use"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.save(save_dir)
    print(f"Full model saved to {save_dir}")
    return save_dir

def load_checkpoint(model, checkpoint_path=None):
    """Load weights from checkpoint"""
    checkpoint_dir = './training_checkpoints_char_rnn_small'
    
    # If specific checkpoint provided, use it
    if checkpoint_path and os.path.exists(checkpoint_path + '.index'):
        print(f"Loading specified checkpoint: {checkpoint_path}")
        model.load_weights(checkpoint_path).expect_partial()
        # Try to extract epoch number from checkpoint path
        try:
            epoch = int(checkpoint_path.split('_')[-1])
            print(f"Loaded weights from epoch {epoch}")
            return model, epoch
        except:
            print("Loaded weights but couldn't determine epoch number")
            return model, 0
    
    # Otherwise look for latest checkpoint
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"Loading latest checkpoint: {latest_checkpoint}")
        try:
            model.load_weights(latest_checkpoint).expect_partial()
            epoch = int(latest_checkpoint.split('_')[-1])
            print(f"Loaded weights from epoch {epoch}")
            return model, epoch
        except Exception as e:
            print(f"Error loading weights: {e}")
            return model, 0
    
    print("No checkpoint found. Using initial random weights.")
    return model, 0

# --- 5. Training Mode ---
if args.mode == 'train':
    print("\n--- TRAINING MODE ---")
    
    # Build training model (batch_size > 1)
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=BATCH_SIZE)
    model.summary()
    
    # Compile the model
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    # Prepare checkpoint directory
    checkpoint_dir = './training_checkpoints_char_rnn_small'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Load weights if checkpoint exists
    model, initial_epoch = load_checkpoint(model, args.checkpoint)
    
    # Training callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    
    # Save information about the vocabulary for generation
    vocab_file = os.path.join("./saved_models", "vocab.npy")
    char_file = os.path.join("./saved_models", "chars.npy")
    
    if not os.path.exists("./saved_models"):
        os.makedirs("./saved_models")
    
    np.save(vocab_file, np.array([char2idx]))
    np.save(char_file, idx2char)
    print(f"Vocabulary mappings saved for generation")
    
    # Train the model
    print(f"\n--- Starting training from epoch {initial_epoch} for {args.epochs} epochs ---")
    start_time = time.time()
    
    history = model.fit(dataset,
                        epochs=initial_epoch + args.epochs,
                        initial_epoch=initial_epoch,
                        callbacks=[checkpoint_callback],
                        verbose=1)
    
    print(f"\nTraining finished. Time taken: {time.time() - start_time:.2f} seconds")
    
    # Save full model for easy reuse
    model_path = save_model(model, "./saved_models/text_generation_model")
    print(f"Full model saved to {model_path}")
    print("\nTo generate text using this model, run:")
    print(f"python main.py --mode generate --checkpoint {tf.train.latest_checkpoint(checkpoint_dir)}")

# --- 6. Generation Mode ---
else:  # args.mode == 'generate'
    print("\n--- GENERATION MODE ---")
    
    # Build generation model (batch_size=1)
    generation_model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    generation_model.build(tf.TensorShape([1, None]))  # Build model with shape [1, None]
    
    # Load weights
    generation_model, _ = load_checkpoint(generation_model, args.checkpoint)
    
    # Try to load vocabulary if needed (for cases where generation is done in a different session)
    vocab_file = os.path.join("./saved_models", "vocab.npy")
    char_file = os.path.join("./saved_models", "chars.npy")
    
    try:
        if os.path.exists(vocab_file) and os.path.exists(char_file):
            saved_char2idx = np.load(vocab_file, allow_pickle=True)[0]
            saved_idx2char = np.load(char_file)
            
            # Only use saved vocab if it matches the current vocab size
            if len(saved_char2idx) == vocab_size:
                print("Using saved vocabulary mappings")
                char2idx = saved_char2idx
                idx2char = saved_idx2char
    except Exception as e:
        print(f"Error loading saved vocabulary: {e}")
    
    def generate_text(model, start_string, num_generate=1000, temperature=1.0):
        print(f"\n--- Generating with seed: '{start_string}' ---")
        print(f"--- Temperature: {temperature} ---")
        
        # Check if all characters in the seed are in our vocabulary
        unknown_chars = [s for s in start_string if s not in char2idx]
        if unknown_chars:
            print(f"WARNING: The following characters in your seed text are not in the vocabulary: {unknown_chars}")
            print("Available characters in vocabulary:", ''.join(sorted(char2idx.keys())))
            
            # Option 1: Replace unknown characters with a known character
            clean_start = ''
            for s in start_string:
                if s in char2idx:
                    clean_start += s
                else:
                    # Replace with first character in vocabulary or space if available
                    replacement = ' ' if ' ' in char2idx else list(char2idx.keys())[0]
                    clean_start += replacement
                    print(f"Replaced '{s}' with '{replacement}'")
            
            if not clean_start:
                # If all characters were unknown, use a default known character
                clean_start = list(char2idx.keys())[0]
                print(f"All characters were unknown. Using '{clean_start}' as seed.")
            
            start_string = clean_start
            print(f"Using modified seed: '{start_string}'")
        
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        model.reset_states()
        current_sequence_indices = input_eval # Start with the seed

        for i in range(num_generate):
            predictions = model(current_sequence_indices)
            predictions = tf.squeeze(predictions, 0)[-1,:] # Get last time step preds
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(tf.expand_dims(predictions, axis=0), num_samples=1)[-1,0].numpy()

            # Add the predicted ID to our sequence for the next input step
            current_sequence_indices = tf.concat([current_sequence_indices, tf.expand_dims([predicted_id], 0)], axis=1)

            # Prevent sequence from getting too long
            if current_sequence_indices.shape[1] > seq_length + 10:
                 current_sequence_indices = current_sequence_indices[:, -seq_length:]

            text_generated.append(idx2char[predicted_id])

        return (start_string + ''.join(text_generated))
    
    # Use a safer default seed that's more likely to be in vocabulary if none specified
    if args.seed != "ROMEO:":
        # Print available characters if custom seed was provided
        print(f"Available characters in vocabulary: '{(''.join(sorted(char2idx.keys())))}'")
        print(f"Attempting to use custom seed: '{args.seed}'")
    
    # Generate text
    print(f"\nGenerating {args.length} characters...")
    generated_output = generate_text(
        generation_model, 
        start_string=args.seed, 
        num_generate=args.length, 
        temperature=args.temp
    )
    
    print("\n--- Generated Text ---")
    print(generated_output)
    
    # Save generated text to file
    output_dir = './generated_text'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = int(time.time())
    output_file = os.path.join(output_dir, f'generated_text_{timestamp}.txt')
    
    with open(output_file, 'w') as f:
        f.write(generated_output)
    
    print(f"\nGenerated text saved to {output_file}")