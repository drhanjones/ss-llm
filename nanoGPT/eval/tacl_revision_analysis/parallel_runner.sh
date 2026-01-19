    #!/bin/bash

    # Parse command line arguments
    if [ $# -lt 3 ]; then
    echo "Usage: $0 <category> <eq_id> <num_parallel>"
    echo "Allowed categories: dundee_rt | naturalstories_sprt"
    echo "Example: $0 dundee_rt 42 10"
        exit 1
    fi

    CATEGORY="$1"
    EQ_ID="$2"
    NUM_PARALLEL="$3"

    # Validate category
    case "$CATEGORY" in
    dundee_rt|naturalstories_sprt)
        ;;
    *)
        echo "Invalid category: $CATEGORY"
        echo "Allowed categories: dundee_rt | naturalstories_sprt"
        exit 1
        ;;
    esac


    # --- CONFIGURATION ---
    SESSION="parallel_runner_${CATEGORY}_eq${EQ_ID}"
    # 1. Set the path to your venv activate script
    VENV_ACTIVATE="/home/athamma/Projects/ss-llm/ss-llm/.venv/bin/activate" 
    # 2. Your python command
    SCRIPT_CMD="python pyscript_reading_type_rpy2_${CATEGORY}.py --equation_pair_index=${EQ_ID}"
    # ---------------------

    # Start the detached session
    tmux new-session -d -s $SESSION

    for i in $(seq 1 "$NUM_PARALLEL")
    do
    tmux new-window -t $SESSION: -n "job-$i"
    
    FULL_CMD="source $VENV_ACTIVATE && clear && $SCRIPT_CMD"
    
    tmux send-keys -t $SESSION:job-$i "$FULL_CMD" C-m
    
    echo "Started job $i with Venv..."
    done

    # Kill the extra initial window created by new-session
    tmux kill-window -t $SESSION:0 2>/dev/null

    echo "✅ $NUM_PARALLEL Jobs running in '$SESSION' using virtualenv."
    echo "   Attach using: tmux attach -t $SESSION"