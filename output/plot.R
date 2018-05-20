library(ggplot2)
library(dplyr)
library(reshape2)

# Somehow change the limits to make graphic readable
trajectorySep <- function(name, trajectory, file){

  sep_trajectory <-  trajectory[trajectory[,1] == name,]
  sep_trajectory$method <- NULL
  sep_trajectory$learning_rate <-  factor(sep_trajectory$learning_rate)
  n <- length(levels(sep_trajectory$learning_rate))
  # Add elseif for different methods to decrease 1 or stay the same
  m <- length(levels(sep_trajectory$decay)) - 1
  title <- capture.output(cat(toupper(name), 'PLOT'))
  filename <- capture.output(cat(file, "_", name, ".jpeg", sep = ""))
  sep_trajectory$x <- rep.int(1:(length(sep_trajectory$value)/(n*m)), n*m)
  colnames(sep_trajectory)[which(colnames(sep_trajectory) == "value")] <- "Val"

  sep_trajectory <- melt(sep_trajectory, id.vars = c("x", "learning_rate", "decay"))
  #print(sep_trajectory)
  
  plt <- ggplot(sep_trajectory, aes(x)) +
    geom_line(aes(y = value, color = variable), size = 1) +
    ggtitle(title) +
    scale_x_continuous(name = 'X', limits = c(0, 100)) +
    scale_y_continuous(name = 'Y') +
    scale_colour_manual(name = "Legend", labels = c("Step size", "Calculated value"), values=c("blue", "red")) + 
    facet_grid(learning_rate ~ decay, scales = "free_y", labeller = label_both)
  ggsave(filename = filename, plot = plt, device = "jpeg", width = 12, height = 12, dpi = 150)
}

trajectoryGraph <- function(file){
  
  trajectory = read.csv(file)
  for (method in levels(factor(trajectory$method)))
       trajectorySep(method, trajectory, sub(".csv", "", file))
}

forEveryFile <- function(dir){
  
  setwd(dir)
  files <- list.files(pattern = "test_*")
  lapply(files, trajectoryGraph)
  setwd("../")
}

setwd("D:/Worknfiles/PyCharm Projects/tf-min-comparison/output")
dirs <-  list.files(pattern = "test10")
lapply(dirs, forEveryFile)



file <- read.csv('test_results.csv')
file <-  file[file[,2] == 'median',]
ggplot(file, aes(iteration)) +
  geom_line(aes(y = value, color = method), size = 1) +
  scale_x_continuous(name = 'X') +
  scale_y_continuous(name = 'Y')



