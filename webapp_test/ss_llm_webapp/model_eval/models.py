from django.db import models

# Create your models here.

class TrainedModel(models.Model):
    """Model to store trained model information."""

    unique_id = models.CharField(max_length=100)
    name = models.CharField(max_length=100)
    description = models.TextField()
    n_layers = models.IntegerField()
    n_heads = models.IntegerField()
    n_embd = models.IntegerField()


    def __str__(self):
        return self.name



