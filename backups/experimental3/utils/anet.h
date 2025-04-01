#include "includes.h"

class utils
{
    public:
        //activation functions
        // * sigmoid    0
        static double sigmoid(double x, bool deriv = false)
        {
            if (deriv)
            {
                return x * (1 - x);
            }
            return 1.0 / (1.0 + exp(-x));
        }
        // * ReLU   1
        static double ReLU(double x, bool deriv = false)
        {
            if (deriv)
            {
                return x > 0 ? 1 : 0;
            }
            return std::max(0.0, x);
        }
        // * SoftPlus   2
        static double softplus(double x, bool deriv = false)
        {
            if(deriv)
            {
                return exp(x) / (1 + exp(x));   
            }
            return log(1 + exp(x));
        }
        double activate(double unactivatedValue, int activationFunction, bool deriv = false)
        {
            switch (activationFunction)
            {
                case 0:
                    return ReLU(unactivatedValue, deriv);
                case 1:
                    return sigmoid(unactivatedValue, deriv);
                case 2:
                    return softplus(unactivatedValue, deriv);
                default:
                    return unactivatedValue;
            }
        }

};


class node
{

    public: 
        struct nextNode
        {
            node* node;
            double weight=1.0;
        };
    private:
        /**
         * *activation functions
         * -1 - none
         * 0 - ReLU
         * 1 - Sigmoid
         * 2 - SoftPlus
        */
        double value = 0.0;
        double unactivatedValue = 0.0;
        double bias = 0.0;
        double delta = 0.0;
        int activationFunction = -1;
        bool end = false; //terminal node on layer

        //next nodes
        std::vector<nextNode> nextNodes;
        
    public:
        node() {}
        node(double value, double bias) 
        {
            this->value = value;
            this->unactivatedValue = value;
            this->bias = bias;
        }
        node(double value, double bias, int activationFunction) 
        {
            this->value = value;
            this->unactivatedValue = value;
            this->bias = bias;
            this->activationFunction = activationFunction;
        }

    

        // * getters and setters
        double getValue() { return value; }
        void setValue(double value) { this->value = value; this->unactivatedValue = value; }

        double getUnactivatedValue() { return unactivatedValue; }
        void setUnactivatedValue(double unactivatedValue) { this->unactivatedValue = unactivatedValue; }

        double getBias() { return bias; }
        void setBias(double bias) { this->bias = bias; }

        double getWeight(int index) { return nextNodes[index].weight; }
        double getWeight(node* node)
        {
            for (int i = 0; i < nextNodes.size(); i++)
            {
                if (nextNodes[i].node == node)
                {
                    return nextNodes[i].weight;
                }
            }
            return 0.0;
        }
        void setWeight(int index, double weight) { nextNodes[index].weight = weight; }
        void setWeight(node* node, double weight)
        {
            for (int i = 0; i < nextNodes.size(); i++)
            {
                if (nextNodes[i].node == node)
                {
                    nextNodes[i].weight = weight;
                    return;
                }
            }
            throw std::invalid_argument("Node not found");
        }



        double getDelta() { return delta; }
        void setDelta(double delta) { this->delta = delta; }

        int getActivationFunction() { return activationFunction; }
        void setActivationFunction(int activationFunction) { this->activationFunction = activationFunction; }

        std::vector<nextNode> getNextNodes() { return nextNodes; }
        void setNextNodes(std::vector<nextNode> nextNodes) { this->nextNodes = nextNodes; }
    
        nextNode getNextNode(int index) { return nextNodes[index]; }
        void removeNextNode(int index) { nextNodes.erase(nextNodes.begin() + index); }

        bool getEnd() { return end; }
        void setEnd(bool end) { this->end = end; }

        // * basic functions

        

        void passValue(node* nextNode, bool end)
        {
            double final;
            nextNode->setUnactivatedValue(
                nextNode->getUnactivatedValue()
                + value * getWeight(nextNode)
                );
            
            if (end)
            {
                final = utils().activate(nextNode->getUnactivatedValue() + nextNode->getBias(), nextNode->getActivationFunction());
                nextNode->setValue(final);
            }
            
        }
        void passValue()
        {
            for (int i = 0; i < nextNodes.size(); i++)
            {
                passValue(nextNodes[i].node, end);
            }
        }


};
