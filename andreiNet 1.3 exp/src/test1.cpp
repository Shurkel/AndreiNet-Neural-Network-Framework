#include "../utils/andreinet.h" // Make sure this points to the Eigen-based net.h
#include <iostream>
#include <vector>
#include <string>
#include <set>        // For vocabulary
#include <map>        // For char-to-index mapping
#include <iomanip>    // For std::setprecision, std::fixed
#include <algorithm>  // For std::shuffle, std::max_element
#include <random>     // For sampling during generation

// --- Configuration ---
const int CONTEXT_LENGTH = 10; // How many previous characters to consider
const int HIDDEN_NODES = 128; // Size of the hidden layer (tune this)
const int EPOCHS = 1000;       // Training epochs (increase for better results, needs time)
const double LEARNING_RATE = 0.05; // Learning rate (tune this)
const int GENERATE_LENGTH = 200; // How many characters to generate after training

// Sample text corpus (same as before)
const std::string corpus = "Alice:" "Hey, have you ever wondered what it would be like to live in space?"

"Bob:" "Oh, definitely! Floating around in zero gravity, seeing Earth from above—it sounds incredible. But I bet it comes with a lot of challenges."

"Alice:" "For sure. Imagine having to exercise for hours just to keep your muscles from weakening. And the food? No fresh pizza up there!"

"Bob:" "No pizza?! Okay, maybe space isn't for me. But think about the adventure—exploring Mars or even going beyond our solar system. The idea of deep space travel is both exciting and terrifying."

"Alice:" "Yeah, it's the unknown that makes it so thrilling. Speaking of the unknown, have you ever had a dream that felt so real you questioned reality when you woke up?"

"Bob:" "Oh man, yes! Just last week, I dreamt I was back in school taking a math test I hadn’t studied for. It felt so real, I actually woke up in a panic."

"Alice:" "Haha, classic stress dream! I get those too. But I once had a dream where I was in a futuristic city, and everything was powered by clean energy. There were floating cars, and AI assistants helped with everything. It was like living in a sci-fi movie."

"Bob:" "That sounds amazing! Speaking of AI, what do you think about chatbots? Some are getting eerily good at conversations."

"Alice:" "I think they're fascinating. The way they process language and learn from interactions is mind-blowing. But sometimes, they lack true understanding—they’re great at imitating conversation but don’t actually ‘think.’"

"Bob:" "Yeah, they don’t have emotions or real experiences. But who knows? Maybe in the future, AI will evolve to a point where we can’t tell the difference."

"Alice:" "That’s both exciting and a little scary. If AI ever becomes truly self-aware, we’d have to rethink what it means to be human."

"Bob:" "True. But let’s switch gears a bit—what’s something totally random you’ve learned recently?"

"Alice:" "Okay, this is weird, but did you know that octopuses have three hearts and their blood is blue because of the copper in it?"

"Bob:" "Whoa! That’s insane. Nature is full of bizarre stuff. Like, did you know that wombat poop is cube-shaped?"

"Alice:" "No way! How does that even happen?"

"Bob:" "Apparently, their intestines have different levels of elasticity, which shapes the poop into cubes. Scientists actually studied it to understand how they do it."

"Alice:" "The things people study never cease to amaze me. But I love how curiosity drives innovation. Even random discoveries can lead to useful inventions."

"Bob:" "Absolutely! Speaking of inventions, if you could create anything, what would it be?"

"Alice:" "Hmm… I think I’d invent a device that lets you record your dreams and watch them later. Imagine how cool that would be!"

"Bob:" "That would be amazing! You could rewatch your best dreams like movies. Though… some nightmares would be terrifying."

"Alice:" "Yeah, maybe we’d need a way to filter out the creepy ones. What about you? What would you invent?"

"Bob:" "I’d love a teleportation device. No more traffic, no more waiting at airports—just instant travel anywhere in the world."

"Alice:" "That would change everything! Though, I bet it would come with some crazy security concerns. What if someone teleports into a bank vault?"

"Bob:" "True. Every new technology has its risks. But imagine the benefits—visiting family instantly, reducing pollution from transportation, even exploring new planets without needing a spaceship."

"Alice:" "You always think big! Speaking of thinking big, do you ever just stare at the stars and wonder about life beyond Earth?"

"Bob:" "All the time. With billions of galaxies out there, it’s hard to believe we’re alone. I just wish we had some real proof."

"Alice:" "Maybe one day. Until then, we can keep dreaming and imagining what’s out there!"

"Bob:" "Agreed. The universe is full of possibilities!";


// --- Helper Functions (Adapted for Eigen) ---

// Create vocabulary and mappings (no change needed)
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

// One-hot encode a character index into an Eigen::VectorXd
Eigen::VectorXd one_hot_encode(int idx, int vocab_size) {
    Eigen::VectorXd vec = Eigen::VectorXd::Zero(vocab_size);
    if (idx >= 0 && idx < vocab_size) {
        vec(idx) = 1.0; // Access elements using () in Eigen
    }
    return vec;
}

// Helper function to find the index of the maximum element (argmax) using Eigen
int argmax(const Eigen::VectorXd& vec) {
    if (vec.size() == 0) {
        return -1;
    }
    Eigen::VectorXd::Index max_idx;
    vec.maxCoeff(&max_idx); // Eigen function to get index of max coefficient
    return static_cast<int>(max_idx);
}


// Sample from probability distribution (output vector) using Eigen
int sample_from_distribution(const Eigen::VectorXd& probabilities, std::mt19937& rng_engine) {
    if (probabilities.size() == 0) return -1;
     // Ensure probabilities are non-negative (can happen with numerical instability)
     Eigen::VectorXd non_negative_probs = probabilities.cwiseMax(0.0);
     double sum = non_negative_probs.sum();
     if (sum <= 0) { // If sum is zero or negative, cannot create distribution
         // Fallback: return argmax or a default index
         return argmax(probabilities); // Use original probabilities for argmax
     }
     // Normalize probabilities (optional but good practice for discrete_distribution)
     // Eigen::VectorXd normalized_probs = non_negative_probs / sum; -> Creates copy
     // Use iterators or pointer directly
     std::discrete_distribution<int> dist(probabilities.data(), probabilities.data() + probabilities.size());
     return dist(rng_engine);
}


// Prepare training data (sliding window) - Adapted for Eigen
TrainingSetEigen prepare_training_data(const std::string& text,
                                   const std::map<char, int>& char_to_idx,
                                   int context_length,
                                   int vocab_size)
{
    TrainingSetEigen data; // Use the Eigen-specific training set type
    if (text.length() <= context_length) {
        std::cerr << "Error: Corpus is too short for the given context length." << std::endl;
        return data;
    }

    int input_vec_size = context_length * vocab_size;

    for (size_t i = 0; i < text.length() - context_length; ++i) {
        InputDataEigen input_sequence(input_vec_size); // Use Eigen type
        input_sequence.setZero(); // Initialize to zero

        // Encode context using Eigen segments
        for (int j = 0; j < context_length; ++j) {
            char context_char = text[i + j];
            int char_idx = char_to_idx.at(context_char);
            Eigen::VectorXd encoded_char = one_hot_encode(char_idx, vocab_size);
            // Place the one-hot vector into the correct segment
            input_sequence.segment(j * vocab_size, vocab_size) = encoded_char;
        }

        // Encode target
        char target_char = text[i + context_length];
        int target_idx = char_to_idx.at(target_char);
        TargetDataEigen target_vector = one_hot_encode(target_idx, vocab_size); // Use Eigen type

        data.push_back({input_sequence, target_vector});
    }
    return data;
}


// --- Main Function ---
int main() {
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
        return 1;
    }

    // 3. Define Network Architecture
    int input_size = CONTEXT_LENGTH * vocab_size;
    int output_size = vocab_size;
    std::vector<int> layerSizes = {input_size, HIDDEN_NODES, output_size};
    Net charNet(layerSizes); // Uses Eigen-based Net constructor

    std::cout << "[+] Created Network Architecture: " << input_size << " -> " << HIDDEN_NODES << " -> " << output_size << std::endl;

    // 4. Configure Network Activations and Loss
    // Access layers directly to set activation functions
    // Input layer (0): Linear (already default in Eigen constructor)
    charNet.layers[1].setActivationFunction(1); // Hidden layer: Sigmoid (ID 1)
    // charNet.layers[1].setActivationFunction(0); // Alternative: ReLU (ID 0)

    charNet.layers[2].setActivationFunction(1); // Output layer: Sigmoid (ID 1) - Paired with CE Loss
                                                // For character generation, Softmax is theoretically better,
                                                // but Sigmoid + Sampling/Argmax works here.

    // Set loss function (Crucial for correct backprop delta calculation)
    charNet.setLossFunction(Net::LossFunction::CROSS_ENTROPY); // Matches Sigmoid output well
    // charNet.setLossFunction(Net::LossFunction::MSE); // Could use MSE, but less common for classification-like tasks

    std::cout << "[+] Network Configuration:" << std::endl;
    charNet.printNetworkStructure(); // Print structure summary


    // 5. Train the Network
    std::cout << "\n[+] Starting Training..." << std::endl;
    std::cout << "    Epochs: " << EPOCHS << std::endl;
    std::cout << "    Learning Rate: " << LEARNING_RATE << std::endl;
    std::cout << "    Training Samples: " << trainingData.size() << std::endl;
    std::cout << "    Batch Size: 1 (SGD)" << std::endl;


    Timer trainingTimer; // Timer from andreinet utils
    trainingTimer.start();

    // Use train function (now takes TrainingSetEigen)
    // Using Batch Size = 1 for Stochastic Gradient Descent (SGD)
    charNet.train(trainingData, EPOCHS, LEARNING_RATE, 1, true);

    trainingTimer.stop(false); // Stop timer without printing built-in message

    std::cout << "\n[+] Training Complete." << std::endl;
    std::cout << "[+] Total Training Time: " << std::fixed << std::setprecision(2)
              << trainingTimer.getDurationMs() << " ms ("
              << trainingTimer.getDurationMs() / 1000.0 << " s)" << std::endl;

    // 6. Generate Text
    std::cout << "\n[+] Generating Text..." << std::endl;

    // Use Util's RNG (via getUtil()) or create a new one
    std::mt19937 gen_rng(std::random_device{}());

    // --- Seed ---
    std::string seed_text = "";
    if (corpus.length() >= CONTEXT_LENGTH) {
        seed_text = corpus.substr(0, CONTEXT_LENGTH); // Start with beginning of corpus
    } else { // Handle short corpus case
         seed_text = std::string(CONTEXT_LENGTH, *vocab.begin()); // Pad with first vocab char
         for(size_t i = 0; i < corpus.length(); ++i) seed_text[i] = corpus[i];
    }

    std::string generated_text = seed_text;
    std::cout << "    Seed: \"" << seed_text << "\"" << std::endl;
    std::cout << "    Generated: " << std::endl << "--------------------------" << std::endl;
    std::cout << generated_text; // Print seed

    std::string current_context = seed_text;
    int input_vec_size = CONTEXT_LENGTH * vocab_size;

    for (int i = 0; i < GENERATE_LENGTH; ++i) {
        // Prepare input vector (Eigen::VectorXd) from current context
        InputDataEigen current_input(input_vec_size); // Eigen type
        current_input.setZero();

        for (int j = 0; j < CONTEXT_LENGTH; ++j) {
             char c = current_context[j];
             int char_idx = -1;
             if(char_to_idx.count(c)) { // Check if char is in vocab
                 char_idx = char_to_idx.at(c);
             } else { // Handle unseen char if necessary (e.g., use a default or skip)
                  char_idx = char_to_idx.at(*vocab.begin()); // Use first vocab char as default
             }
             Eigen::VectorXd encoded_char = one_hot_encode(char_idx, vocab_size);
             current_input.segment(j * vocab_size, vocab_size) = encoded_char;
        }


        // Predict next character probabilities
        // predict() returns a const reference to the output layer's activation vector
        const Eigen::VectorXd& output_probabilities = charNet.predict(current_input);

        // --- Choose Next Character ---
        // Option 1: Argmax (deterministic, often leads to loops)
        // int predicted_idx = argmax(output_probabilities);

        // Option 2: Sampling (more diverse, uses probability distribution)
         int predicted_idx = sample_from_distribution(output_probabilities, gen_rng);

        char next_char = '?'; // Default if index is bad
        if (idx_to_char.count(predicted_idx)) {
            next_char = idx_to_char.at(predicted_idx);
        } else if (output_probabilities.size() > 0) {
             // Fallback if sampling gave weird index? Maybe try argmax
             predicted_idx = argmax(output_probabilities);
             if (idx_to_char.count(predicted_idx)) next_char = idx_to_char.at(predicted_idx);
        }


        // Append and update context
        generated_text += next_char;
        std::cout << next_char << std::flush; // Print char immediately

        // Slide context window: take last (CONTEXT_LENGTH - 1) chars and append new char
        if (current_context.length() >= CONTEXT_LENGTH){
             current_context = current_context.substr(1);
        }
        current_context += next_char;
    }

    std::cout << std::endl << "--------------------------" << std::endl;
    std::cout << "[+] Generation Finished." << std::endl;

    return 0;
}