library(ggplot2)
trends = read.csv('trend_0_10_0.csv')

n = 10

adam_trends = trends[trends[,1] == 'adam',]
ggplot(adam_trends, aes(x = rep.int(1:(length(VALUE)/n), n), y = VALUE, color = as.factor(LR))) +
  geom_line(size = 1, alpha = 0.75) +
  geom_point() + 
  ggtitle('ADAM PLOT') +
  scale_color_discrete(name = "Learning rates") +
  scale_x_continuous(name = 'X',
                     limits = c(0, length(adam_trends[,3])/n)) + 
  scale_y_continuous(name = 'Y',
                     limits = c(min(adam_trends[,3]), max(adam_trends[,3]))) +
  facet_grid(LR~.)



momentum_trends = trends[trends[,1] == 'momentum',]
ggplot(momentum_trends, aes(x = rep.int(1:(length(VALUE)/n), n), y = VALUE, color = as.factor(LR))) +
  geom_line(size = 1, alpha = 0.75) +
  geom_point() + 
  ggtitle('MOMENTUM PLOT') +
  scale_color_discrete(name = "Learning rates") +
  scale_x_continuous(name = 'X',
                     limits = c(0, length(momentum_trends[,3])/n)) + 
  scale_y_continuous(name = 'Y',
                     limits = c(min(momentum_trends[,3]), max(momentum_trends[,3]))) +
  facet_grid(LR~.)



adadelta_trends = trends[trends[,1] == 'adadelta',]
ggplot(adadelta_trends, aes(x = rep.int(1:(length(VALUE)/n), n), y = VALUE, color = as.factor(LR))) +
  geom_line(size = 1, alpha = 0.75) +
  geom_point() + 
  ggtitle('ADADELTA PLOT') +
  scale_color_discrete(name = "Learning rates") +
  scale_x_continuous(name = 'X',
                     limits = c(0, length(adadelta_trends[,3])/n)) + 
  scale_y_continuous(name = 'Y',
                     limits = c(min(adadelta_trends[,3]), max(adadelta_trends[,3]))) +
  facet_grid(LR~.)



adagrad_trends = trends[trends[,1] == 'adagrad',]
ggplot(adagrad_trends, aes(x = rep.int(1:(length(VALUE)/n), n), y = VALUE, color = as.factor(LR))) +
  geom_line(size = 1, alpha = 0.75) +
  geom_point() + 
  ggtitle('ADAGRAD PLOT') +
  scale_color_discrete(name = "Learning rates") +
  scale_x_continuous(name = 'X',
                     limits = c(0, length(adagrad_trends[,3])/n)) + 
  scale_y_continuous(name = 'Y',
                     limits = c(min(adagrad_trends[,3]), max(adagrad_trends[,3]))) +
  facet_grid(LR~.)
