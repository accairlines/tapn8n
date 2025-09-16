from django.core.management.base import BaseCommand
import gc
import psutil
import os

class Command(BaseCommand):
    help = 'Clear memory and reset database connections'

    def handle(self, *args, **options):
        # Force garbage collection
        collected = gc.collect()
        self.stdout.write(f'Garbage collected {collected} objects')
        
        # Get memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.stdout.write(f'Current memory usage: {memory_mb:.2f} MB')
        
        # Reset database connections
        from django.db import connections
        connections.close_all()
        self.stdout.write('Database connections closed')
        
        self.stdout.write(
            self.style.SUCCESS('Memory cleanup completed successfully')
        )
