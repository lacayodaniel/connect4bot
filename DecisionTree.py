import numpy as np

class DecisionTree(object):
    """This class contains the main object used throughout this project: a decision tree. It contains methods
    to visualise and evaluate the trees."""

    def __init__(self, right=None, left=None, label='', value=None):
        """Create a node of a decision tree"""
        self.right = right
        '''right child, taken when a sample[`decisiontree.DecisionTree.label`] > `decisiontree.DecisionTree.value`'''
        self.left = left
        '''left child, taken when sample[`decisiontree.DecisionTree.label`] <= `decisiontree.DecisionTree.value`'''
        self.label = label
        '''string representation of the attribute the node splits on'''
        self.value = value
        '''the value where the node splits on (if `None`, then we're in a leaf)'''

    def evaluate(self, feature_vector):
        """Create a prediction for a sample (using its feature vector)

        **Params**
        ----------
          - `feature_vector` (pandas Series or dict) - the sample to evaluate, must be a `pandas Series` object or a
          `dict`. It is important that the attribute keys in the sample are the same as the labels occuring in the tree.

        **Returns**
        -----------
            the predicted class label
        """
        if self.value is None:
            return self.label
        else:
            # feature_vector should only contain 1 row
            if feature_vector[self.label] < self.value:
                return self.left.evaluate(feature_vector)
            else:
                return self.right.evaluate(feature_vector)

    def serialize(self):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        def serializeHelper(node):
            if node.value is not None:
                vals.append(str(node.label)+'<'+str(int(np.floor(node.value) + 1)))
                serializeHelper(node.left)
                serializeHelper(node.right)
            else:
                vals.append(str(node.label))
        vals = []
        serializeHelper(self)
        return ' '.join(vals)

    @staticmethod
    def deserialize(tree_string):
        nodes = tree_string.split()
        nr_nodes = len(nodes)
        def deserializeHelper(counter):
            if counter < nr_nodes:
                node = str(nodes[counter])
                if '<' in node:
                    label, value = node.split('<')
                    dt_node = DecisionTree(label=int(label), value=int(value))

                    node, counter = deserializeHelper(counter + 1)
                    if node is not None:
                        dt_node.left = node

                    node, counter = deserializeHelper(counter)
                    if node is not None:
                        dt_node.right = node

                    return dt_node, counter
                else:
                    return DecisionTree(label=node), counter + 1
            else:
                return None, counter
        #print('RESULT:', deserializeHelper(0)[0])
        return deserializeHelper(0)[0]
