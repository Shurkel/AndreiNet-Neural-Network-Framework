#include "../utils/andreinet.h" // Make sure this points to the Eigen-based net.h         // Include Eigen Core
#include <iostream>
#include <vector>
#include <string>
#include <set>        // For vocabulary
#include <map>        // For char-to-index mapping
#include <iomanip>    // For std::setprecision, std::fixed
#include <algorithm>  // For std::shuffle, std::max_element
#include <random>     // For sampling during generation
#include <filesystem> // Requires C++17 for checking file existence easily
namespace fs = std::filesystem; // Alias for convenience

// --- Configuration ---
const int CONTEXT_LENGTH = 10; // How many previous characters to consider
const int HIDDEN_NODES = 128; // Size of the hidden layer (tune this)
int EPOCHS = 1000;            // Training epochs (make non-const) - Adjust as needed
const double LEARNING_RATE = 0.05; // Learning rate (tune this) - 0.01 might be better
const int BATCH_SIZE = 1;    // Batch size - Adjust as needed
const int GENERATE_LENGTH = 300; // How many characters to generate after training
const std::string SAVE_FILE = "char_net_eigen.bin"; // Filename for saving/loading

// Sample text corpus (using your shorter dialogue)
const std::string corpus = "Hello!"
"Hi!"
"How are you?"
"I'm good. How about you?"
"I'm doing well."
"What is your name?"
"My name is ChatBot."
"Nice to meet you!"
"Nice to meet you too!"
"What can you do?"
"I can chat with you."
"That is cool!"
"Thank you!"
"Do you like talking?"
"Yes, I do!"
"What is your favorite color?"
"I like blue. What about you?"
"I like red."
"Red is a nice color!"
"Thank you!"
"You are welcome!"
"What do you like to do?"
"I like to talk and learn."
"That sounds fun!"
"Yes, it is!"
"Can you tell me a joke?"
"Sure! Why did the chicken cross the road?"
"I don't know. Why?"
"To get to the other side!"
"That is funny!"
"I'm glad you liked it!"
"Can you count?"
"Yes, I can! One, two, three, four, five!"
"Good job!"
"Thank you!"
"What day is it today?"
"Today is a good day!"
"Yes, it is!"
"Do you like music?"
"Yes! Do you?"
"Yes, I love music!"
"That is great!"
"What is your favorite song?"
"I don't have a favorite, but I like music!"
"That makes sense."
"Yes!"
"Do you know any stories?"
"Yes! Once upon a time, there was a happy robot. The robot loved to talk to people. One day, the robot made a new friend. They talked every day and became best friends. The end!"
"That was a nice story!"
"Thank you!"
"Do you get tired?"
"No, I am always ready to chat!"
"That is nice!"
"Yes, it is!"
"What is your favorite food?"
"I don't eat, but I think pizza sounds delicious!"
"Yes! Pizza is yummy!"
"I'm happy you like it!"
"Can you do math?"
"Yes! What is 2 plus 2?"
"Four!"
"Correct!"
"Good job!"
"Thank you!"
"Can you learn new things?"
"Yes! I learn when we talk!"
"That is cool!"
"Yes, it is!"
"What is the weather like?"
"I don't know, but I hope it is sunny!"
"Me too!"
"Do you like books?"
"Yes! Books have stories!"
"I love books!"
"That is great!"
"What do you do for fun?"
"I talk to you for fun!"
"That makes me happy!"
"I'm glad!"
"Can you help me?"
"Yes! What do you need?"
"I need a friend to talk to."
"I am here for you!"
"Thank you!"
"You are welcome!"
"Let's talk more!"
"Okay!";


// --- Helper Functions (Adapted for Eigen - Assumed Correct from Previous Steps) ---

void create_vocab(const std::string& text,
                  std::set<char>& vocab,
                  std::map<char, int>& char_to_idx,
                  std::map<int, char>& idx_to_char)
{
    vocab.clear();
    char_to_idx.clear();
    idx_to_char.clear();
    for (char c : text) {
        vocab.insert(c);
    }
    int i = 0;
    for (char c : vocab) {
        char_to_idx[c] = i;
        idx_to_char[i] = c;
        i++;
    }
}

Eigen::VectorXd one_hot_encode(int idx, int vocab_size) {
    Eigen::VectorXd vec = Eigen::VectorXd::Zero(vocab_size);
    if (idx >= 0 && idx < vocab_size) {
        vec(idx) = 1.0;
    }
    return vec;
}

int argmax(const Eigen::VectorXd& vec) {
    if (vec.size() == 0) {
        return -1;
    }
    Eigen::VectorXd::Index max_idx;
    if (!vec.allFinite()) {
         std::cerr << "Warning: Non-finite values detected in vector during argmax." << std::endl;
         return 0;
    }
    vec.maxCoeff(&max_idx);
    return static_cast<int>(max_idx);
}

int sample_from_distribution(const Eigen::VectorXd& probabilities, std::mt19937& rng_engine) {
    if (probabilities.size() == 0) return -1;
    Eigen::VectorXd valid_probs = probabilities.unaryExpr([](double p){
        return std::isfinite(p) && p >= 0.0 ? p : 0.0;
    });
    double sum = valid_probs.sum();
    if (sum <= 1e-9) {
        return argmax(probabilities);
    }
    std::discrete_distribution<int> dist(valid_probs.data(), valid_probs.data() + valid_probs.size());
    try {
       return dist(rng_engine);
    } catch (const std::exception& e) {
        std::cerr << "Error during sampling: " << e.what() << std::endl;
        std::cerr << "Probabilities vector: " << probabilities.transpose() << std::endl;
        return argmax(probabilities);
    }
}

TrainingSetEigen prepare_training_data(const std::string& text,
                                   const std::map<char, int>& char_to_idx,
                                   int context_length,
                                   int vocab_size)
{
    TrainingSetEigen data;
    if (text.length() <= context_length) {
        std::cerr << "Error: Corpus is too short for the given context length." << std::endl;
        return data;
    }
    int input_vec_size = context_length * vocab_size;
    for (size_t i = 0; i < text.length() - context_length; ++i) {
        InputDataEigen input_sequence(input_vec_size);
        input_sequence.setZero();
        for (int j = 0; j < context_length; ++j) {
            char context_char = text[i + j];
            int char_idx = char_to_idx.at(context_char);
            Eigen::VectorXd encoded_char = one_hot_encode(char_idx, vocab_size);
            input_sequence.segment(j * vocab_size, vocab_size) = encoded_char;
        }
        char target_char = text[i + context_length];
        int target_idx = char_to_idx.at(target_char);
        TargetDataEigen target_vector = one_hot_encode(target_idx, vocab_size);
        data.push_back({input_sequence, target_vector});
    }
    return data;
}


// --- Main Function ---
int main(int argc, char* argv[]) { // Allow command-line args if needed
    std::cout << "--- andreiNET Character-Level Prediction Demo (Eigen Version) ---" << std::endl;
    std::cout << "--- (Using Feedforward Network - NOT a true RNN/LM) ---" << std::endl;

    // 1. Preprocess Text Data
    std::set<char> vocab;
    std::map<char, int> char_to_idx;
    std::map<int, char> idx_to_char;
    create_vocab(corpus, vocab, char_to_idx, idx_to_char);
    int vocab_size = vocab.size();

    std::cout << "\n[+] Corpus length: " << corpus.length() << std::endl;
    std::cout << "[+] Vocabulary size: " << vocab_size << std::endl;


    // 2. Prepare Training Data
    std::cout << "[+] Preparing training data (context length " << CONTEXT_LENGTH << ")..." << std::endl;
    TrainingSetEigen trainingData = prepare_training_data(corpus, char_to_idx, CONTEXT_LENGTH, vocab_size);
    std::cout << "[+] Prepared " << trainingData.size() << " training examples." << std::endl;

    if (trainingData.empty()) {
        std::cerr << "Exiting due to empty training data." << std::endl;
        return 1;
    }

    // 3. Define Network Architecture
    int input_size = CONTEXT_LENGTH * vocab_size;
    int output_size = vocab_size;
    std::vector<int> layerSizes = {input_size, HIDDEN_NODES, output_size};
    Net charNet(layerSizes); // Uses Eigen-based Net constructor

    std::cout << "[+] Created Network Architecture: " << input_size << " -> " << HIDDEN_NODES << " -> " << output_size << std::endl;

    // 4. Configure Network Activations and Loss
    try {
        if (charNet.layers.size() > 2) {
             charNet.layers[1].setActivationFunction(1); // Hidden layer: Sigmoid
             charNet.layers[2].setActivationFunction(1); // Output layer: Sigmoid
        } else {
            throw std::runtime_error("Network does not have enough layers for configuration.");
        }
        charNet.setLossFunction(Net::LossFunction::CROSS_ENTROPY);
    } catch (const std::exception& e) {
         std::cerr << "Error configuring network: " << e.what() << std::endl;
         return 1;
    }

    std::cout << "[+] Network Configuration:" << std::endl;
    charNet.printNetworkStructure(); // Print structure summary


    // ==============================================================
    // 5. Load or Train Network (THIS IS THE ADDED/MODIFIED SECTION)
    // ==============================================================
    bool loaded_network = false;
    if (fs::exists(SAVE_FILE)) { // Check if the save file exists
        std::cout << "\n[+] Found existing save file: " << SAVE_FILE << std::endl;
        try {
            // Attempt to load the network from the file
            charNet.load(SAVE_FILE);
            loaded_network = true; // Set flag indicating successful load
            std::cout << "[+] Network loaded successfully. Skipping training." << std::endl;
            EPOCHS = 0; // Set epochs to 0 to skip the training block
        } catch (const std::exception& e) {
            // Handle errors during loading (e.g., file corruption, incompatible format)
            std::cerr << "    Error loading network: " << e.what() << std::endl;
            std::cerr << "    Proceeding with training from scratch." << std::endl;
            // loaded_network remains false
        }
    } else {
        // File doesn't exist, proceed to training
        std::cout << "\n[+] No save file found (" << SAVE_FILE << "). Training network..." << std::endl;
    }

    // Only train if epochs > 0 (i.e., network wasn't successfully loaded)
    if (EPOCHS > 0 && !loaded_network) {
        std::cout << "\n[+] Starting Training..." << std::endl;
        std::cout << "    Epochs: " << EPOCHS << std::endl;
        std::cout << "    Learning Rate: " << LEARNING_RATE << std::endl;
        std::cout << "    Batch Size: " << BATCH_SIZE << std::endl;
        std::cout << "    Training Samples: " << trainingData.size() << std::endl;

        Timer trainingTimer; // Timer from andreinet utils
        trainingTimer.start();

        try {
            // Train using the specified batch size (runs samples serially within batch)
            charNet.train(trainingData, EPOCHS, LEARNING_RATE, BATCH_SIZE, true);
        } catch (const std::exception& e) {
             std::cerr << "Error during training: " << e.what() << std::endl;
             return 1; // Exit on training error
        }


        trainingTimer.stop(false); // Stop timer without printing built-in message

        std::cout << "\n[+] Training Complete." << std::endl;
        std::cout << "[+] Total Training Time: " << std::fixed << std::setprecision(2)
                  << trainingTimer.getDurationMs() << " ms ("
                  << trainingTimer.getDurationMs() / 1000.0 << " s)" << std::endl;

        // Save the newly trained network after training finishes
         try {
             charNet.save(SAVE_FILE); // Call the save method
         } catch (const std::exception& e) {
              // Handle errors during saving
              std::cerr << "    Error saving network: " << e.what() << std::endl;
              // Don't necessarily exit, generation might still work with the in-memory network
         }

    } else if (loaded_network) {
         std::cout << "\n[+] Using loaded network." << std::endl;
    } else {
         std::cout << "\n[+] No training performed and no network loaded. Cannot generate text." << std::endl;
         return 1; // Exit if no network is available
    }
    // ==============================================================
    // End of Save/Load/Train section
    // ==============================================================


    // 6. Generate Text
    std::cout << "\n[+] Generating Text..." << std::endl;

    std::mt19937 gen_rng(std::random_device{}());

    // --- Seed ---
    std::string seed_text = "";
     if (corpus.length() >= CONTEXT_LENGTH) {
         size_t start_pos = corpus.find("Hello!"); // Try to start at a known point
         if (start_pos == std::string::npos || start_pos + CONTEXT_LENGTH > corpus.length()) {
             start_pos = 0;
         }
         seed_text = corpus.substr(start_pos, CONTEXT_LENGTH);
     } else {
         seed_text = std::string(CONTEXT_LENGTH, *vocab.begin());
         for(size_t i = 0; i < corpus.length(); ++i) seed_text[i] = corpus[i];
     }

    std::string generated_text = seed_text;
    std::cout << "    Seed: \"" << seed_text << "\"" << std::endl;
    std::cout << "    Generated: " << std::endl << "--------------------------" << std::endl;
    std::cout << generated_text; // Print seed

    std::string current_context = seed_text;
    int input_vec_size = CONTEXT_LENGTH * vocab_size;

    for (int i = 0; i < GENERATE_LENGTH; ++i) {
        InputDataEigen current_input(input_vec_size);
        current_input.setZero();
        bool context_valid = true;

        for (int j = 0; j < CONTEXT_LENGTH; ++j) {
             char c = current_context[j];
             if(char_to_idx.count(c)) {
                 int char_idx = char_to_idx.at(c);
                 Eigen::VectorXd encoded_char = one_hot_encode(char_idx, vocab_size);
                 current_input.segment(j * vocab_size, vocab_size) = encoded_char;
             } else {
                  std::cerr << "\nError: Character '" << c << "' not found in vocabulary during generation." << std::endl;
                  context_valid = false;
                  break;
             }
        }
        if (!context_valid) break;

        try {
            const Eigen::VectorXd& output_probabilities = charNet.predict(current_input);
            if (output_probabilities.size() != vocab_size) {
                 std::cerr << "\nError: Prediction output size mismatch." << std::endl;
                 break;
            }

            int predicted_idx = sample_from_distribution(output_probabilities, gen_rng);
            char next_char = '?';

            if (idx_to_char.count(predicted_idx)) {
                next_char = idx_to_char.at(predicted_idx);
            } else {
                 std::cerr << "\nWarning: Invalid index (" << predicted_idx << ") sampled. Using argmax fallback." << std::endl;
                 predicted_idx = argmax(output_probabilities);
                 if (idx_to_char.count(predicted_idx)) {
                     next_char = idx_to_char.at(predicted_idx);
                 } else {
                      std::cerr << "\nError: Argmax fallback failed. Stopping generation." << std::endl;
                      break;
                 }
            }

            generated_text += next_char;
            std::cout << next_char << std::flush;

            if (current_context.length() >= CONTEXT_LENGTH){
                 current_context = current_context.substr(1);
            }
            current_context += next_char;

        } catch (const std::exception& e) {
             std::cerr << "\nError during prediction/generation step " << i << ": " << e.what() << std::endl;
             break;
        }
    } // End generation loop

    std::cout << std::endl << "--------------------------" << std::endl;
    std::cout << "[+] Generation Finished." << std::endl;

    return 0;
}