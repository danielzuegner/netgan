"""
Implementation of the method proposed in the paper:
'Adversarial Attacks on Classification Models for Graphs'
by Aleksandar Bojchevski, Oleksandr Shchur, Daniel Zügner, Stephan Günnemann
Published at ICML 2018 in Stockholm, Sweden.

Copyright (C) 2018
Daniel Zügner
Technical University of Munich
"""

import tensorflow as tf
from netgan import utils
import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import os
from matplotlib import pyplot as plt

class NetGAN:
    """
    NetGAN class, an implicit generative model for graphs using random walks.
    """

    def __init__(self, N, rw_len, walk_generator, generator_layers=[40], discriminator_layers=[30],
                 W_down_generator_size=128, W_down_discriminator_size=128, batch_size=128, noise_dim=16,
                 noise_type="Gaussian", learning_rate=0.0003, disc_iters=3, wasserstein_penalty=10,
                 l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5, temp_start=5.0, min_temperature=0.5,
                 temperature_decay=1-5e-5, seed=15, gpu_id=0, use_gumbel=True, legacy_generator=False):
        """
        Initialize NetGAN.

        Parameters
        ----------
        N: int
           Number of nodes in the graph to generate.
        rw_len: int
                Length of random walks to generate.
        walk_generator: function
                        Function that generates a single random walk and takes no arguments.
        generator_layers: list of integers, default: [40], i.e. a single layer with 40 units.
                          The layer sizes of the generator LSTM layers
        discriminator_layers: list of integers, default: [30], i.e. a single layer with 30 units.
                              The sizes of the discriminator LSTM layers
        W_down_generator_size: int, default: 128
                               The size of the weight matrix W_down of the generator. See our paper for details.
        W_down_discriminator_size: int, default: 128
                                   The size of the weight matrix W_down of the discriminator. See our paper for details.
        batch_size: int, default: 128
                    The batch size.
        noise_dim: int, default: 16
                   The dimension of the random noise that is used as input to the generator.
        noise_type: str in ["Gaussian", "Uniform], default: "Gaussian"
                    The noise type to feed into the generator.
        learning_rate: float, default: 0.0003
                       The learning rate.
        disc_iters: int, default: 3
                    The number of discriminator iterations per generator training iteration.
        wasserstein_penalty: float, default: 10
                             The Wasserstein gradient penalty applied to the discriminator. See the Wasserstein GAN
                             paper for details.
        l2_penalty_generator: float, default: 1e-7
                                L2 penalty on the generator weights.
        l2_penalty_discriminator: float, default: 5e-5
                                    L2 penalty on the discriminator weights.
        temp_start: float, default: 5.0
                    The initial temperature for the Gumbel softmax.
        min_temperature: float, default: 0.5
                         The minimal temperature for the Gumbel softmax.
        temperature_decay: float, default: 1-5e-5
                           After each evaluation, the current temperature is updated as
                           current_temp := max(temperature_decay*current_temp, min_temperature)
        seed: int, default: 15
              Random seed.
        gpu_id: int or None, default: 0
                The ID of the GPU to be used for training. If None, CPU only.
        use_gumbel: bool, default: True
                Use the Gumbel softmax trick.
        
        legacy_generator: bool, default: False
            If True, the hidden and cell states of the generator LSTM are initialized by two separate feed-forward networks. 
            If False (recommended), the hidden layer is shared, which has less parameters and performs just as good.
        
        """

        self.params = {
            'noise_dim': noise_dim,
            'noise_type': noise_type,
            'Generator_Layers': generator_layers,
            'Discriminator_Layers': discriminator_layers,
            'W_Down_Generator_size': W_down_generator_size,
            'W_Down_Discriminator_size': W_down_discriminator_size,
            'l2_penalty_generator': l2_penalty_generator,
            'l2_penalty_discriminator': l2_penalty_discriminator,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'Wasserstein_penalty': wasserstein_penalty,
            'temp_start': temp_start,
            'min_temperature': min_temperature,
            'temperature_decay': temperature_decay,
            'disc_iters': disc_iters,
            'use_gumbel': use_gumbel,
            'legacy_generator': legacy_generator
        }

        assert rw_len > 1, "Random walk length must be > 1."

        tf.set_random_seed(seed)

        self.N = N
        self.rw_len = rw_len

        self.noise_dim = self.params['noise_dim']
        self.G_layers = self.params['Generator_Layers']
        self.D_layers = self.params['Discriminator_Layers']
        self.tau = tf.placeholder(1.0 , shape=(), name="temperature")

        # W_down and W_up for generator and discriminator
        self.W_down_generator = tf.get_variable('Generator.W_Down',
                                                shape=[self.N, self.params['W_Down_Generator_size']],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())

        self.W_down_discriminator = tf.get_variable('Discriminator.W_Down',
                                                    shape=[self.N, self.params['W_Down_Discriminator_size']],
                                                    dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer())

        self.W_up = tf.get_variable("Generator.W_up", shape = [self.G_layers[-1], self.N],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

        self.b_W_up = tf.get_variable("Generator.W_up_bias", dtype=tf.float32, initializer=tf.zeros_initializer,
                                      shape=self.N)

        self.generator_function = self.generator_recurrent
        self.discriminator_function = self.discriminator_recurrent

        self.fake_inputs = self.generator_function(self.params['batch_size'], reuse=False, gumbel=use_gumbel, legacy=legacy_generator)
        self.fake_inputs_discrete = self.generate_discrete(self.params['batch_size'], reuse=True,
                                                           gumbel=use_gumbel, legacy=legacy_generator)

        # Pre-fetch real random walks
        dataset = tf.data.Dataset.from_generator(walk_generator, tf.int32, [self.params['batch_size'], self.rw_len])
        #dataset_batch = dataset.prefetch(2).batch(self.params['batch_size'])
        dataset_batch = dataset.prefetch(100)
        batch_iterator = dataset_batch.make_one_shot_iterator()
        real_data = batch_iterator.get_next()

        self.real_inputs_discrete = real_data
        self.real_inputs = tf.one_hot(self.real_inputs_discrete, self.N)

        self.disc_real = self.discriminator_function(self.real_inputs)
        self.disc_fake = self.discriminator_function(self.fake_inputs, reuse=True)

        self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
        self.gen_cost = -tf.reduce_mean(self.disc_fake)

        # WGAN lipschitz-penalty
        alpha = tf.random_uniform(
            shape=[self.params['batch_size'], 1, 1],
            minval=0.,
            maxval=1.
        )

        self.differences = self.fake_inputs - self.real_inputs
        self.interpolates = self.real_inputs + (alpha * self.differences)
        self.gradients = tf.gradients(self.discriminator_function(self.interpolates, reuse=True), self.interpolates)[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1, 2]))
        self.gradient_penalty = tf.reduce_mean((self.slopes - 1.) ** 2)
        self.disc_cost += self.params['Wasserstein_penalty'] * self.gradient_penalty

        # weight regularization; we omit W_down from regularization
        self.disc_l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                     if 'Disc' in v.name
                                     and not 'W_down' in v.name]) * self.params['l2_penalty_discriminator']
        self.disc_cost += self.disc_l2_loss

        # weight regularization; we omit  W_down from regularization
        self.gen_l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                     if 'Gen' in v.name
                                     and not 'W_down' in v.name]) * self.params['l2_penalty_generator']
        self.gen_cost += self.gen_l2_loss

        self.gen_params = [v for v in tf.trainable_variables() if 'Generator' in v.name]
        self.disc_params = [v for v in tf.trainable_variables() if 'Discriminator' in v.name]

        self.gen_train_op = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate'], beta1=0.5,
                                                   beta2=0.9).minimize(self.gen_cost, var_list=self.gen_params)
        self.disc_train_op = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate'], beta1=0.5,
                                                    beta2=0.9).minimize(self.disc_cost, var_list=self.disc_params)

        if gpu_id is None:
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
        else:
            gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
            config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.InteractiveSession(config=config)
        self.init_op = tf.global_variables_initializer()


    def generate_discrete(self, n_samples, reuse=True, z=None, gumbel=True, legacy=False):
        """
        Generate a random walk in index representation (instead of one hot). This is faster but prevents the gradients
        from flowing into the generator, so we only use it for evaluation purposes.

        Parameters
        ----------
        n_samples: int
                   The number of random walks to generate.
        reuse: bool, default: None
               If True, generator variables will be reused.
        z: None or tensor of shape (n_samples, noise_dim)
           The input noise. None means that the default noise generation function will be used.
        gumbel: bool, default: False
            Whether to use the gumbel softmax for generating discrete output.
        legacy: bool, default: False
            If True, the hidden and cell states of the generator LSTM are initialized by two separate feed-forward networks. 
            If False (recommended), the hidden layer is shared, which has less parameters and performs just as good.
        
        Returns
        -------
                The generated random walks, shape [None, rw_len, N]


        """

        return tf.argmax(self.generator_function(n_samples, reuse, z, gumbel=gumbel, legacy=legacy), axis=-1)

    def generator_recurrent(self, n_samples, reuse=None, z=None, gumbel=True, legacy=False):
        """
        Generate random walks using LSTM.
        Parameters
        ----------
        n_samples: int
                   The number of random walks to generate.
        reuse: bool, default: None
               If True, generator variables will be reused.
        z: None or tensor of shape (n_samples, noise_dim)
           The input noise. None means that the default noise generation function will be used.
        gumbel: bool, default: False
            Whether to use the gumbel softmax for generating discrete output.
        legacy: bool, default: False
            If True, the hidden and cell states of the generator LSTM are initialized by two separate feed-forward networks. 
            If False (recommended), the hidden layer is shared, which has less parameters and performs just as good.
        Returns
        -------
        The generated random walks, shape [None, rw_len, N]

        """

        with tf.variable_scope('Generator') as scope:
            if reuse is True:
                scope.reuse_variables()

            def lstm_cell(lstm_size):
                return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)

            self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.G_layers])

            # initial states h and c are randomly sampled for each lstm cell
            if z is None:
                initial_states_noise = make_noise([n_samples, self.noise_dim], self.params['noise_type'])
            else:
                initial_states_noise = z
            initial_states = []

            # Noise preprocessing
            for ix,size in enumerate(self.G_layers):
                if legacy: # old version to initialize LSTM. new version has less parameters and performs just as good.
                    h_intermediate = tf.layers.dense(initial_states_noise, size, name="Generator.h_int_{}".format(ix+1),
                                                     reuse=reuse, activation=tf.nn.tanh)
                    h = tf.layers.dense(h_intermediate, size, name="Generator.h_{}".format(ix+1), reuse=reuse,
                                        activation=tf.nn.tanh)

                    c_intermediate = tf.layers.dense(initial_states_noise, size, name="Generator.c_int_{}".format(ix+1),
                                                     reuse=reuse, activation=tf.nn.tanh)
                    c = tf.layers.dense(c_intermediate, size, name="Generator.c_{}".format(ix+1), reuse=reuse,
                                        activation=tf.nn.tanh)
                    
                else:
                    intermediate = tf.layers.dense(initial_states_noise, size, name="Generator.int_{}".format(ix+1),
                                                     reuse=reuse, activation=tf.nn.tanh)
                    h = tf.layers.dense(intermediate, size, name="Generator.h_{}".format(ix+1), reuse=reuse,
                                        activation=tf.nn.tanh)
                    c = tf.layers.dense(intermediate, size, name="Generator.c_{}".format(ix+1), reuse=reuse,
                                        activation=tf.nn.tanh)
                initial_states.append((c, h))

            state = initial_states
            inputs = tf.zeros([n_samples, self.params['W_Down_Generator_size']])
            outputs = []

            # LSTM tine steps
            for i in range(self.rw_len):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                # Get LSTM output
                output, state = self.stacked_lstm.call(inputs, state)

                # Blow up to dimension N using W_up
                output_bef = tf.matmul(output, self.W_up) + self.b_W_up

                # Perform Gumbel softmax to ensure gradients flow
                if gumbel:
                    output = gumbel_softmax(output_bef, temperature=self.tau, hard = True)
                else:
                    output = tf.nn.softmax(output_bef)

                # Back to dimension d
                inputs = tf.matmul(output, self.W_down_generator)

                outputs.append(output)
            outputs = tf.stack(outputs, axis=1)
        return outputs

    def discriminator_recurrent(self, inputs, reuse=None):
        """
        Discriminate real from fake random walks using LSTM.
        Parameters
        ----------
        inputs: tf.tensor, shape (None, rw_len, N)
                The inputs to process
        reuse: bool, default: None
               If True, discriminator variables will be reused.

        Returns
        -------
        final_score: tf.tensor, shape [None,], i.e. a scalar
                     A score measuring how "real" the input random walks are perceived.

        """

        with tf.variable_scope('Discriminator') as scope:
            if reuse == True:
                scope.reuse_variables()

            input_reshape = tf.reshape(inputs, [-1, self.N])
            output = tf.matmul(input_reshape, self.W_down_discriminator)
            output = tf.reshape(output, [-1, self.rw_len, int(self.W_down_discriminator.get_shape()[-1])])

            def lstm_cell(lstm_size):
                return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)

            disc_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.D_layers])

            output_disc, state_disc = tf.contrib.rnn.static_rnn(cell=disc_lstm_cell, inputs=tf.unstack(output, axis=1),
                                                              dtype='float32')

            last_output = output_disc[-1]

            final_score = tf.layers.dense(last_output, 1, reuse=reuse, name="Discriminator.Out")
            return final_score

    def train(self, A_orig, val_ones, val_zeros,  max_iters=50000, stopping=None, eval_transitions=15e6,
              transitions_per_iter=150000, max_patience=5, eval_every=500, plot_every=-1, save_directory="../snapshots",
              model_name=None, continue_training=False):
        """

        Parameters
        ----------
        A_orig: sparse matrix, shape: (N,N)
                Adjacency matrix of the original graph to be trained on.
        val_ones: np.array, shape (n_val, 2)
                  The indices of the hold-out set of validation edges
        val_zeros: np.array, shape (n_val, 2)
                  The indices of the hold-out set of validation non-edges
        max_iters: int, default: 50,000
                   The maximum number of training iterations if early stopping does not apply.
        stopping: float in (0,1] or None, default: None
                  The early stopping strategy. None means VAL criterion will be used (i.e. evaluation on the
                  validation set and stopping after there has not been an improvement for *max_patience* steps.
                  Set to a value in the interval (0,1] to stop when the edge overlap exceeds this threshold.
        eval_transitions: int, default: 15e6
                          The number of transitions that will be used for evaluating the validation performance, e.g.
                          if the random walk length is 5, each random walk contains 4 transitions.
        transitions_per_iter: int, default: 150000
                              The number of transitions that will be generated in one batch. Higher means faster
                              generation, but more RAM usage.
        max_patience: int, default: 5
                      Maximum evaluation steps without improvement of the validation accuracy to tolerate. Only
                      applies to the VAL criterion.
        eval_every: int, default: 500
                    Evaluate the model every X iterations.
        plot_every: int, default: -1
                    Plot the generator/discriminator losses every X iterations. Set to None or a negative number
                           to disable plotting.
        save_directory: str, default: "../snapshots"
                        The directory to save model snapshots to.
        model_name: str, default: None
                    Name of the model (will be used for saving the snapshots).
        continue_training: bool, default: False
                           Whether to start training without initializing the weights first. If False, weights will be
                           initialized.

        Returns
        -------
        log_dict: dict
                  A dictionary with the following values observed during training:
                  * The generator and discriminator losses
                  * The validation performances (ROC and AP)
                  * The edge overlap values between the generated and original graph
                  * The sampled graphs for all evaluation steps.

        """

        if stopping == None:  # use VAL criterion
            best_performance = 0.0
            patience = max_patience
            print("**** Using VAL criterion for early stopping ****")

        else:  # use EO criterion
            assert "float" in str(type(stopping)) and stopping > 0 and stopping <= 1
            print("**** Using EO criterion of {} for early stopping".format(stopping))

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        if model_name is None:
            # Find the file corresponding to the lowest vacant model number to store the snapshots into.
            model_number = 0
            while os.path.exists("{}/model_best_{}.ckpt".format(save_directory, model_number)):
                model_number += 1
            save_file = "{}/model_best_{}.ckpt".format(save_directory, model_number)
            open(save_file, 'a').close()  # touch file
        else:
            save_file = "{}/{}_best.ckpt".format(save_directory, model_name)
        print("**** Saving snapshots into {} ****".format(save_file))

        if not continue_training:
            print("**** Initializing... ****")
            self.session.run(self.init_op)
            print("**** Done.           ****")
        else:
            print("**** Continuing training without initializing weights. ****")

        # Validation labels
        actual_labels_val = np.append(np.ones(len(val_ones)), np.zeros(len(val_zeros)))

        # Some lists to store data into.
        gen_losses = []
        disc_losses = []
        graphs = []
        val_performances = []
        eo=[]
        temperature = self.params['temp_start']

        starting_time = time.time()
        saver = tf.train.Saver()

        transitions_per_walk = self.rw_len - 1
        # Sample lots of random walks, used for evaluation of model.
        sample_many_count = int(np.round(transitions_per_iter/transitions_per_walk))
        sample_many = self.generate_discrete(sample_many_count, reuse=True)
        n_eval_walks = eval_transitions/transitions_per_walk
        n_eval_iters = int(np.round(n_eval_walks/sample_many_count))

        print("**** Starting training. ****")

        for _it in range(max_iters):

            if _it > 0 and _it % (2500) == 0:
                t = time.time() - starting_time
                print('{:<7}/{:<8} training iterations, took {} seconds so far...'.format(_it, max_iters, int(t)))

            # Generator training iteration
            gen_loss, _ = self.session.run([self.gen_cost, self.gen_train_op],
                                           feed_dict={self.tau: temperature})

            _disc_l = []
            # Multiple discriminator training iterations.
            for _ in range(self.params['disc_iters']):
                disc_loss, _ = self.session.run(
                    [self.disc_cost, self.disc_train_op],
                    feed_dict={self.tau: temperature}
                )
                _disc_l.append(disc_loss)

            gen_losses.append(gen_loss)
            disc_losses.append(np.mean(_disc_l))

            # Evaluate the model's progress.
            if _it > 0 and _it % eval_every == 0:

                # Sample lots of random walks.
                smpls = []
                for _ in range(n_eval_iters):
                    smpls.append(self.session.run(sample_many, {self.tau: 0.5}))

                # Compute score matrix
                gr = utils.score_matrix_from_random_walks(np.array(smpls).reshape([-1, self.rw_len]), self.N)
                gr = gr.tocsr()

                # Assemble a graph from the score matrix
                _graph = utils.graph_from_scores(gr, A_orig.sum())
                # Compute edge overlap
                edge_overlap = utils.edge_overlap(A_orig.toarray(), _graph)
                graphs.append(_graph)
                eo.append(edge_overlap)

                edge_scores = np.append(gr[tuple(val_ones.T)].A1, gr[tuple(val_zeros.T)].A1)

                # Compute Validation ROC-AUC and average precision scores.
                val_performances.append((roc_auc_score(actual_labels_val, edge_scores),
                                               average_precision_score(actual_labels_val, edge_scores)))

                # Update Gumbel temperature
                temperature = np.maximum(self.params['temp_start'] * np.exp(-(1-self.params['temperature_decay']) * _it),
                                         self.params['min_temperature'])

                print("**** Iter {:<6} Val ROC {:.3f}, AP: {:.3f}, EO {:.3f} ****".format(_it,
                                                                               val_performances[-1][0],
                                                                               val_performances[-1][1],
                                                                               edge_overlap/A_orig.sum()))

                if stopping is None:   # Evaluate VAL criterion
                    if np.sum(val_performances[-1]) > best_performance:
                        # New "best" model
                        best_performance = np.sum(val_performances[-1])
                        patience = max_patience
                        _ = saver.save(self.session, save_file)
                    else:
                        patience -= 1

                    if patience == 0:
                        print("**** EARLY STOPPING AFTER {} ITERATIONS ****".format(_it))
                        break
                elif edge_overlap/A_orig.sum() >= stopping:   # Evaluate EO criterion
                    print("**** EARLY STOPPING AFTER {} ITERATIONS ****".format(_it))
                    break
                    
            if plot_every > 0 and (_it+1) % plot_every == 0:
                if len(disc_losses) > 10:
                    plt.plot(disc_losses[9::], label="Critic loss")
                    plt.plot(gen_losses[9::], label="Generator loss")
                else:
                    plt.plot(disc_losses, label="Critic loss")
                    plt.plot(gen_losses, label="Generator loss")
                plt.legend()
                plt.show()

        print("**** Training completed after {} iterations. ****".format(_it))
        plt.plot(disc_losses[9::], label="Critic loss")
        plt.plot(gen_losses[9::], label="Generator loss")
        plt.legend()
        plt.show()
        if stopping is None:
            saver.restore(self.session, save_file)
        #### Training completed.
        log_dict = {"disc_losses": disc_losses, 'gen_losses': gen_losses, 'val_performances': val_performances,
                    'edge_overlaps': eo, 'generated_graphs': graphs}
        return log_dict


def make_noise(shape, type="Gaussian"):
    """
    Generate random noise.

    Parameters
    ----------
    shape: List or tuple indicating the shape of the noise
    type: str, "Gaussian" or "Uniform", default: "Gaussian".

    Returns
    -------
    noise tensor

    """

    if type == "Gaussian":
        noise = tf.random_normal(shape)
    elif type == 'Uniform':
        noise = tf.random_uniform(shape, minval=-1, maxval=1)
    else:
        print("ERROR: Noise type {} not supported".format(type))
    return noise


def sample_gumbel(shape, eps=1e-20):
    """
    Sample from a uniform Gumbel distribution. Code by Eric Jang available at
    http://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    Parameters
    ----------
    shape: Shape of the Gumbel noise
    eps: Epsilon for numerical stability.

    Returns
    -------
    Noise drawn from a uniform Gumbel distribution.

    """
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
      """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y