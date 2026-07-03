#### Assumption: we are running in ~/scratch/OptimalSGT/

HOME_DIR="$HOME/scratch/OptimalSGT"
FARM1="$HOME_DIR/farm1" # ~/scratch/OptimalSGT/farm1/table.dat
FARM1_TABLE="farm1/table.dat"


### FARM 1 SETUP ### 
FARM1_COUNT=0
# clear everything in FARM1_TABLE
> $FARM1_TABLE

DATASETS=(
    adult avila bank bean bidding electricity eucalyptus eye-movements eye-state fault gas-drift htru magic mini-boone mushroom occupancy page pendigits raisin rice room segment skin wilt
)


### ------------------------------- Generalization Experiments ------------------------------- ###

# for FILE in branches_run.py # literati_run.py cart_run.py dpdt_run.py shapecart_run.py shapetao_run.py axtao_run.py  contree_run.py 
# do
# for dataset in "${DATASETS[@]}" 
# do
# for i in $(seq 1 200)
# do
# for fold in 0 1 2 3 4
# do
#     # write to FARM1_TABLE
#     ((FARM1_COUNT++))
#     echo "$FARM1_COUNT python $HOME_DIR/runners/$FILE --dataset $dataset --fold $fold --trial-id $i --home-dir $HOME_DIR" >> $FARM1_TABLE 
# done
# done
# done
# done


## ------------------------------- Anytime Experiments ------------------------------- ###

# for dataset in avila fault page rice eye-movements
# do 
# for fold in $(seq 0 4)
# do
        # ----------------------------------------------------------------------------------------
        # # anytime AO* with cart and round robin selection
        # ((FARM1_COUNT++)) 
        # echo "$FARM1_COUNT python $HOME_DIR/runners/anytime_literati.py --dataset $dataset --fold $fold --home-dir $HOME_DIR --heuristic cart --anytime" >> $FARM1_TABLE
        
        # # anytime AO* with impurity and round robin selection
        # ((FARM1_COUNT++)) 
        # echo "$FARM1_COUNT python $HOME_DIR/runners/anytime_literati.py --dataset $dataset --fold $fold --home-dir $HOME_DIR  --heuristic impurity --anytime" >> $FARM1_TABLE
        
        # # anytime AO* with primal and round robin selection
        # ((FARM1_COUNT++)) 
        # echo "$FARM1_COUNT python $HOME_DIR/runners/anytime_literati.py --dataset $dataset --fold $fold --home-dir $HOME_DIR  --heuristic primal --anytime" >> $FARM1_TABLE 

        # # anytime AO* with cart and bfs selection # THIS IS EQUIVALENT TO BRANCHES SELECTION WITH CART HEURISTIC
        # ((FARM1_COUNT++)) 
        # echo "$FARM1_COUNT python $HOME_DIR/runners/anytime_literati.py --dataset $dataset --fold $fold --home-dir $HOME_DIR --heuristic cart --anytime --selection_scheme bfs" >> $FARM1_TABLE
        
        # # anytime AO* with impurity and bfs selection # THIS IS EQUIVALENT TO BRANCHES SELECTION WITH IMPURITY HEURISTIC
        # ((FARM1_COUNT++)) 
        # echo "$FARM1_COUNT python $HOME_DIR/runners/anytime_literati.py --dataset $dataset --fold $fold --home-dir $HOME_DIR  --heuristic impurity --anytime --selection_scheme bfs" >> $FARM1_TABLE
        
        # # anytime AO* with primal and bfs selection
        # ((FARM1_COUNT++)) 
        # echo "$FARM1_COUNT python $HOME_DIR/runners/anytime_literati.py --dataset $dataset --fold $fold --home-dir $HOME_DIR  --heuristic primal --anytime --selection_scheme bfs" >> $FARM1_TABLE 
        
        # # Anytime AO* with cart heuristic and branches (primal) selection
        # ((FARM1_COUNT++)) 
        # echo "$FARM1_COUNT python $HOME_DIR/runners/anytime_literati.py --dataset $dataset --fold $fold --home-dir $HOME_DIR  --heuristic cart --selection_scheme branches" >> $FARM1_TABLE

        # ----------------------------------------------------------------------------------------

        # # native AO* with bfs selection
        # ((FARM1_COUNT++)) 
        # echo "$FARM1_COUNT python $HOME_DIR/runners/anytime_literati.py --dataset $dataset --fold $fold --home-dir $HOME_DIR  --heuristic dual --selection_scheme bfs" >> $FARM1_TABLE
        
        # # native AO* with branches selection
        # ((FARM1_COUNT++)) 
        # echo "$FARM1_COUNT python $HOME_DIR/runners/anytime_literati.py --dataset $dataset --fold $fold --home-dir $HOME_DIR  --heuristic dual --selection_scheme branches" >> $FARM1_TABLE

        # # native AO* with round-robin selection
        # ((FARM1_COUNT++)) 
        # echo "$FARM1_COUNT python $HOME_DIR/runners/anytime_literati.py --dataset $dataset --fold $fold --home-dir $HOME_DIR  --heuristic dual --selection_scheme round_robin" >> $FARM1_TABLE

        # # native AO* with bfs selection and cart tie-breaking
        # ((FARM1_COUNT++)) 
        # echo "$FARM1_COUNT python $HOME_DIR/runners/anytime_literati.py --dataset $dataset --fold $fold --home-dir $HOME_DIR  --heuristic cart --selection_scheme bfs" >> $FARM1_TABLE

        # # native AO* with round-robin selection and cart tie-breaking
        # ((FARM1_COUNT++)) 
        # echo "$FARM1_COUNT python $HOME_DIR/runners/anytime_literati.py --dataset $dataset --fold $fold --home-dir $HOME_DIR  --heuristic cart --selection_scheme round_robin" >> $FARM1_TABLE
# done
# done


# ------------------------------- Optimality Experiments ------------------------------- ###
for dataset in "${DATASETS[@]}" 
do
for fold in $(seq 0 4)
do
for depth in $(seq 3 6)
do
    # # optimality experiments for ConTree
    # ((FARM1_COUNT++)) 
    # echo "$FARM1_COUNT python $HOME_DIR/runners/optimality_contree_run.py --dataset $dataset --fold $fold --home-dir $HOME_DIR --depth $depth" >> $FARM1_TABLE

    # # optimality experiments for CART
    # ((FARM1_COUNT++)) 
    # echo "$FARM1_COUNT python $HOME_DIR/runners/optimality_cart_run.py --dataset $dataset --fold $fold --home-dir $HOME_DIR --depth $depth" >> $FARM1_TABLE

    # # optimality experiments for ShapeCART
    # ((FARM1_COUNT++)) 
    # echo "$FARM1_COUNT python $HOME_DIR/runners/optimality_shapecart_run.py --dataset $dataset --fold $fold --home-dir $HOME_DIR  --depth $depth" >> $FARM1_TABLE

    # # optimality experiments for ShapeTAO
    # ((FARM1_COUNT++)) 
    # echo "$FARM1_COUNT python $HOME_DIR/runners/optimality_shapetao_run.py --dataset $dataset --fold $fold --home-dir $HOME_DIR  --depth $depth" >> $FARM1_TABLE

    # # optimality experiments for AxTAO
    # ((FARM1_COUNT++)) 
    # echo "$FARM1_COUNT python $HOME_DIR/runners/optimality_axtao_run.py --dataset $dataset --fold $fold --home-dir $HOME_DIR  --depth $depth" >> $FARM1_TABLE

    # # optimality experiments for STreeD + STreeD-Light
    # ((FARM1_COUNT++)) 
    # echo "$FARM1_COUNT python $HOME_DIR/runners/optimality_streed_run.py --dataset $dataset --fold $fold --home-dir $HOME_DIR --depth $depth" >> $FARM1_TABLE
    # ((FARM1_COUNT++)) 
    # echo "$FARM1_COUNT python $HOME_DIR/runners/optimality_streed_run.py --dataset $dataset --fold $fold --home-dir $HOME_DIR --light --depth $depth" >> $FARM1_TABLE
    
    # optimality experiments for Branches + Branches-Light
    ((FARM1_COUNT++))
    echo "$FARM1_COUNT python -u $HOME_DIR/runners/optimality_branches_run.py --dataset $dataset --fold $fold --home-dir $HOME_DIR --depth $depth" >> $FARM1_TABLE
    # ((FARM1_COUNT++))
    # echo "$FARM1_COUNT python $HOME_DIR/runners/optimality_branches_run.py --dataset $dataset --fold $fold --home-dir $HOME_DIR --light --depth $depth" >> $FARM1_TABLE

    # # optimality experiments for DPDT + DPDT-Light
    # ((FARM1_COUNT++)) 
    # echo "$FARM1_COUNT python $HOME_DIR/runners/optimality_dpdt_run.py --dataset $dataset --fold $fold --home-dir $HOME_DIR --depth $depth" >> $FARM1_TABLE
    # ((FARM1_COUNT++)) 
    # echo "$FARM1_COUNT python $HOME_DIR/runners/optimality_dpdt_run.py --dataset $dataset --fold $fold --home-dir $HOME_DIR --light --depth $depth" >> $FARM1_TABLE

    # # optimality experiments for Literati + Literati-Light
    # for k in $(seq 1 3)
    # do 
    #     ((FARM1_COUNT++)) 
    #     echo "$FARM1_COUNT python $HOME_DIR/runners/optimality_literati_run.py --dataset $dataset --fold $fold --home-dir $HOME_DIR --max_order $k  --depth $depth" >> $FARM1_TABLE
    #     ((FARM1_COUNT++)) 
    #     echo "$FARM1_COUNT python $HOME_DIR/runners/optimality_literati_run.py --dataset $dataset --fold $fold --home-dir $HOME_DIR --light --max_order $k --depth $depth" >> $FARM1_TABLE
    # done
done
done
done

## ------------------------------- PSweep Experiments ------------------------------- ###
# for dataset in avila fault page rice eye-movements
# do
# for fold in $(seq 0 4)
# do
# for depth in 6
# do
# for trial_id in $(seq 0 49)
# do
# for k in $(seq 1 3)
# do 
#     ((FARM1_COUNT++)) 
#     echo "$FARM1_COUNT python $HOME_DIR/runners/psweep_literati.py --dataset $dataset --trial_id $trial_id --fold $fold --home-dir $HOME_DIR --light --max_order $k --depth $depth" >> $FARM1_TABLE
# done
# done
# done
# done
# done

# remove any trailing empty lines from FARM1_TABLE
sed -i -e :a -e '/^\s*$/d;N;ba' $FARM1_TABLE
echo "Total FARM1 jobs: $FARM1_COUNT"