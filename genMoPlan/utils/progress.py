import time
import math


def format_time(seconds):
    """
    Format seconds into HH:MM:SS format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string in HH:MM:SS format
    """
    if seconds is None:
        return "N/A"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ETAIterator:
    """
    Iterator wrapper that provides ETA calculation.
    
    This class wraps any iterator and tracks elapsed time to estimate
    the remaining time to completion.
    """
    
    def __init__(self, iterator, total):
        """
        Initialize an ETAIterator.
        
        Args:
            iterator: The iterator to wrap
            total: The total number of iterations expected
        """
        self.iterator = iterator
        self.total = total
        self.current = 0
        self.start_time = None
        self._eta = None
        
    def __iter__(self):
        """Return self as iterator"""
        self.start_time = time.time()
        return self
        
    def __next__(self):
        """Get the next item and update ETA"""
        try:
            if self.current > 0:
                elapsed = time.time() - self.start_time
                iterations_done = self.current
                iterations_left = self.total - self.current
                
                seconds_per_iter = elapsed / iterations_done
                self._eta = seconds_per_iter * iterations_left

            item = next(self.iterator)
            self.current += 1
            
            return item
        except StopIteration:
            raise
            
    @property
    def eta(self):
        """Get the ETA in seconds"""
        return self._eta
        
    @property
    def eta_formatted(self):
        """Get the ETA formatted as HH:MM:SS"""
        return format_time(self._eta)
        
    @property
    def progress(self):
        """Get the current progress as a fraction (0.0 to 1.0)"""
        if self.total > 0:
            return self.current / self.total
        return 0
        
    @property
    def progress_percent(self):
        """Get the current progress as a percentage"""
        return self.progress * 100


class Progress:

    def __init__(
        self,
        total,
        name="Progress",
        ncol=3,
        max_length=20,
        indent=0,
        line_width=100,
        speed_update_freq=100,
    ):
        self.total = total
        self.name = name
        self.ncol = ncol
        self.max_length = max_length
        self.indent = indent
        self.line_width = line_width
        self._speed_update_freq = speed_update_freq

        self._step = 0
        self._prev_line = "\033[F"
        self._clear_line = " " * self.line_width

        self._pbar_size = self.ncol * self.max_length
        self._complete_pbar = "#" * self._pbar_size
        self._incomplete_pbar = " " * self._pbar_size

        self.lines = [""]
        self.fraction = "{} / {}".format(0, self.total)

        self.resume()

    def update(self, description="", n=1):
        self._step += n
        if self._step % self._speed_update_freq == 0:
            self._time0 = time.time()
            self._step0 = self._step
        self.set_description(description)

    def resume(self):
        self._skip_lines = 1
        print("\n", end="")
        self._time0 = time.time()
        self._step0 = self._step

    def pause(self):
        self._clear()
        self._skip_lines = 1

    def set_description(self, params=[]):

        if type(params) == dict:
            params = sorted([(key, val) for key, val in params.items()])

        ############
        # Position #
        ############
        self._clear()

        ###########
        # Percent #
        ###########
        percent, fraction = self._format_percent(self._step, self.total)
        self.fraction = fraction

        #########
        # Speed #
        #########
        speed = self._format_speed(self._step)

        ##########
        # Params #
        ##########
        num_params = len(params)
        nrow = math.ceil(num_params / self.ncol)
        params_split = self._chunk(params, self.ncol)
        params_string, lines = self._format(params_split)
        self.lines = lines

        description = "{} | {}{}".format(percent, speed, params_string)
        print(description)
        self._skip_lines = nrow + 1

    def append_description(self, descr):
        self.lines.append(descr)

    def _clear(self):
        position = self._prev_line * self._skip_lines
        empty = "\n".join([self._clear_line for _ in range(self._skip_lines)])
        print(position, end="")
        print(empty)
        print(position, end="")

    def _format_percent(self, n, total):
        if total:
            percent = n / float(total)

            complete_entries = int(percent * self._pbar_size)
            incomplete_entries = self._pbar_size - complete_entries

            pbar = (
                self._complete_pbar[:complete_entries]
                + self._incomplete_pbar[:incomplete_entries]
            )
            fraction = "{} / {}".format(n, total)
            string = "{} [{}] {:3d}%".format(fraction, pbar, int(percent * 100))
        else:
            fraction = "{}".format(n)
            string = "{} iterations".format(n)
        return string, fraction

    def _format_speed(self, n):
        num_steps = n - self._step0
        t = time.time() - self._time0
        speed = num_steps / t
        string = "{:.1f} Hz".format(speed)
        if num_steps > 0:
            self._speed = string
        return string

    def _chunk(self, l, n):
        return [l[i : i + n] for i in range(0, len(l), n)]

    def _format(self, chunks):
        lines = [self._format_chunk(chunk) for chunk in chunks]
        lines.insert(0, "")
        padding = "\n" + " " * self.indent
        string = padding.join(lines)
        return string, lines

    def _format_chunk(self, chunk):
        line = " | ".join([self._format_param(param) for param in chunk])
        return line

    def _format_param(self, param):
        k, v = param
        return "{} : {}".format(k, v)[: self.max_length]

    def stamp(self):
        if self.lines != [""]:
            params = " | ".join(self.lines)
            string = "[ {} ] {}{} | {}".format(
                self.name, self.fraction, params, self._speed
            )
            self._clear()
            print(string, end="\n")
            self._skip_lines = 1
        else:
            self._clear()
            self._skip_lines = 0

    def close(self):
        self.pause()


class Silent:

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        return lambda *args: None


if __name__ == "__main__":
    silent = Silent()
    silent.update()
    silent.stamp()

    num_steps = 1000
    progress = Progress(num_steps)
    for i in range(num_steps):
        progress.update()
        params = [
            ["A", "{:06d}".format(i)],
            ["B", "{:06d}".format(i)],
            ["C", "{:06d}".format(i)],
            ["D", "{:06d}".format(i)],
            ["E", "{:06d}".format(i)],
            ["F", "{:06d}".format(i)],
            ["G", "{:06d}".format(i)],
            ["H", "{:06d}".format(i)],
        ]
        progress.set_description(params)
        time.sleep(0.01)
    progress.close()
