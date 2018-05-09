# plotting example distributions of degrees of uncertainty
# using artificial data
library("ggplot2")

setwd("/home/loki/Nextcloud/Uni/MSc Thesis/latex/images/")

x <- seq(0, 1, length=1000)
y1 <- dnorm(x, mean=0, sd=.005) + dnorm(x, mean=1, sd=.005)
y2 <- dnorm(x, mean=.1, sd=.1) + dnorm(x, mean=.9, sd=.1)
y3 <- dnorm(x, mean=.5, sd=.2)

# plot(x, dnorm(x, mean=0, sd=.005) + dnorm(x, mean=1, sd=.005),
#      type="l", lwd=1)
df.ex1 <- data.frame(x=x, y=y1)
df.ex2 <- data.frame(x=x, y=y2)
df.ex3 <- data.frame(x=x, y=y3)

g1 <- ggplot(data = df.ex1) +
  geom_line(aes(x=x, y=y)) + xlab("") + ylab("") +
  theme_minimal() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  scale_x_continuous(breaks=c(0,1))
ggsave("uncertain_1.png", width=5, height=4, units="cm")

g2 <- ggplot(data = df.ex2) +
  geom_line(aes(x=x, y=y)) + xlab("") + ylab("") +
  geom_vline(xintercept = c(.1, .9), linetype="dotted") +
  theme_minimal() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  scale_x_continuous(breaks=c(0,1))
ggsave("uncertain_2.png", width=5, height=4, units="cm")
  
g3 <- ggplot(data = df.ex3) +
 geom_line(aes(x=x, y=y)) + xlab("") + ylab("") +
  geom_vline(xintercept = c(.5), linetype="dotted") +
  theme_minimal() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  scale_x_continuous(breaks=c(0,1))
ggsave("uncertain_3.png", width=5, height=4, units="cm")