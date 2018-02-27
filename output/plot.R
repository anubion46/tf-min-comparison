library(ggplot2)
trends = read.csv('trend_0_10_0.csv')

n = 6

adam_trends = trends[trends[,1] == 'adam',]
adam_trends$learning_rate = factor(adam_trends$learning_rate)

ggplot(adam_trends, aes(x = rep.int(1:(length(value)/n), n), y = value, color = as.factor(learning_rate))) +
  geom_line(size = 1, alpha = 0.75) +
  geom_point() + 
  ggtitle('ADAM PLOT') +
  scale_color_discrete(name = "Learning rates") +
  scale_x_continuous(name = 'X',
                     limits = c(0, length(adam_trends[,3])/n)) + 
  scale_y_continuous(name = 'Y',
                     limits = c(min(adam_trends[,3]), max(adam_trends[,3]))) +
  facet_grid(learning_rate~.)



momentum_trends = trends[trends[,1] == 'momentum',]
momentum_trends$learning_rate = factor(momentum_trends$learning_rate)

ggplot(momentum_trends, aes(x = rep.int(1:(length(value)/n), n), y = value, color = as.factor(learning_rate))) +
  geom_line(size = 1, alpha = 0.75) +
  geom_point() + 
  ggtitle('MOMENTUM PLOT') +
  scale_color_discrete(name = "Learning rates") +
  scale_x_continuous(name = 'X',
                     limits = c(0, length(momentum_trends[,3])/n)) + 
  scale_y_continuous(name = 'Y',
                     limits = c(min(momentum_trends[,3]), max(momentum_trends[,3]))) +
  facet_grid(learning_rate~.)



adadelta_trends = trends[trends[,1] == 'adadelta',]
adadelta_trends$learning_rate = factor(adadelta_trends$learning_rate)

ggplot(adadelta_trends, aes(x = rep.int(1:(length(value)/n), n), y = value, color = as.factor(learning_rate))) +
  geom_line(size = 1, alpha = 0.75) +
  geom_point() + 
  ggtitle('ADADELTA PLOT') +
  scale_color_discrete(name = "Learning rates") +
  scale_x_continuous(name = 'X',
                     limits = c(0, length(adadelta_trends[,3])/n)) + 
  scale_y_continuous(name = 'Y',
                     limits = c(min(adadelta_trends[,3]), max(adadelta_trends[,3]))) +
  facet_grid(learning_rate~.)



adagrad_trends = trends[trends[,1] == 'adagrad',]
adagrad_trends$learning_rate = factor(adagrad_trends$learning_rate)

ggplot(adagrad_trends, aes(x = rep.int(1:(length(value)/n), n), y = value, color = as.factor(learning_rate))) +
  geom_line(size = 1, alpha = 0.75) +
  geom_point() + 
  ggtitle('ADAGRAD PLOT') +
  scale_color_discrete(name = "Learning rates") +
  scale_x_continuous(name = 'X',
                     limits = c(0, length(adagrad_trends[,3])/n)) + 
  scale_y_continuous(name = 'Y',
                     limits = c(min(adagrad_trends[,3]), max(adagrad_trends[,3]))) +
  facet_grid(learning_rate~.)
