import tensorflow as tf
import numpy as np
import gym

##===========================================================================
##==============================The neural network============================
##===========================================================================
num_input = 4 # there are 4 observation (like speed, angles, etc)
num_hidden = 20 #dimension of the hidden_layer
num_outputs = 1 # prob to go left
learning_rate = 0.01
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None,num_input])

hidden_layer  = tf.layers.dense(X,num_hidden,activation = tf.nn.elu,kernel_initializer=initializer)
#tf.layers.dropout(inputs=hidden_layer,rate=0.8)

#hidden_layer  = tf.layers.dense(hidden_layer,num_hidden*2,activation = tf.nn.elu,kernel_initializer=initializer)
#tf.layers.dropout(inputs=hidden_layer,rate=0.8)

#hidden_layer  = tf.layers.dense(hidden_layer,num_hidden*4,activation = tf.nn.elu,kernel_initializer=initializer)
#tf.layers.dropout(inputs=hidden_layer,rate=0.8)

#hidden_layer  = tf.layers.dense(hidden_layer,num_hidden*2,activation = tf.nn.elu,kernel_initializer=initializer)
#tf.layers.dropout(inputs=hidden_layer,rate=0.8)

hidden_layer  = tf.layers.dense(hidden_layer,num_hidden,activation = tf.nn.elu,kernel_initializer=initializer)
#tf.layers.dropout(inputs=hidden_layer,rate=0.8)

logits = tf.layers.dense(hidden_layer,num_outputs)
output = tf.nn.sigmoid(logits)

prob = tf.concat(axis=1,values=[output,1-output])
action = tf.multinomial(prob,num_samples=1) # this is the action done, left or right [L,R]

y = 1.0 - tf.to_float(action)

#optimizer
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)

gradients_and_variables = optimizer.compute_gradients(cross_entropy)


gradients = []
gradient_placeholders = []
grads_and_vars_feed = []

for gradient, variable in gradients_and_variables:
    gradients.append(gradient)
    gradient_placeholder = tf.placeholder(tf.float32, shape=gradient.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))

#apply the calculated gradient with the discount
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()

## function to calculate the rewards discount

def helper_discount_rewards(rewards, discount_rate):
    '''
    Takes in rewards and applies discount rate
    '''
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    '''
    Takes in all rewards, applies helper_discount function and then normalizes
    using mean and std.
    '''
    all_discounted_rewards = []
    for rewards in all_rewards:
        all_discounted_rewards.append(helper_discount_rewards(rewards,discount_rate))

    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

##===========================================================================
##=====================Play the game/ Train the model=========================
##===========================================================================


num_game_rounds = 1
num_episodes = 1
discount_rate = 0.95

env = gym.make('CartPole-v1')

with tf.Session() as sess:
    sess.run(init)

    mean_score = []
    max_score = []

    for episode in range(num_episodes):

        all_rewards = []
        all_gradients = []
        current_score = []

        for game in range(num_game_rounds): # fait pleins de game
            current_rewards = []
            current_gradients = []
            observations = env.reset()
            done = False
            steps=0
            while not done:
                action_val, gradient_val = sess.run([action,gradients],feed_dict={X:observations.reshape(1,num_input)})
                observations, reward, done, info  = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradient_val)
                steps += 1

            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)
            current_score.append(steps)

        mean_score.append(np.mean(current_score))
        max_score.append(np.max(current_score))

        print("on episode: {}, max score = {} mean_score = {} ".format(episode, max_score[episode],mean_score[episode] ))


        all_rewards = discount_and_normalize_rewards(all_rewards,discount_rate)
        feed_dict = {}

        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients


        sess.run(training_op,feed_dict=feed_dict)


#############################################
### RUN TRAINED MODEL ON ENVIRONMENT #######
###########################################

#############################################
### RUN TRAINED MODEL ON ENVIRONMENT #######
###########################################


for game in range(10):
    observations = env.reset()
    with tf.Session() as sess:

        # https://www.tensorflow.org/api_guides/python/meta_graph
        new_saver = tf.train.import_meta_graph('./model/my-extreme-step-model.meta')
        new_saver.restore(sess,'./model/my-extreme-step-model')
        i=0
        while True:
            env.render()
            action_val, gradients_val = sess.run([action, gradients], feed_dict={X: observations.reshape(1, num_input)})
            observations, reward, done, info = env.step(action_val[0][0])
            i+=1
            if done:
                print(i)
                break
