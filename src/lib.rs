#[allow(unused_imports)]
use std::{
    collections::hash_map::Entry,
    collections::{HashMap, HashSet},
    ops::Range,
    str::FromStr,
};

pub fn load_file(filename: &str) -> String {
    std::fs::read_to_string(format!("inputs/{filename}"))
        .unwrap_or_else(|_err| panic!("Failed to read {}", filename))
}

pub enum Part {
    Part1,
    Part2,
}
pub use Part::*;

#[allow(unused_macros)]
macro_rules! v {
    (@iter $e:expr, for $p:pat in $l:expr $(, if $f:expr)?) => {{
        #[allow(unused_variables)]
        let x = $l.into_iter() $(.filter(|$p| $f))? .map(|$p| $e); x
    }};
    (@set $($rest:tt)*) => { v!(@iter $($rest)*).collect::<HashSet<_>>() };
    (     $($rest:tt)*) => { v!(@iter $($rest)*).collect::<Vec<_>>() };
}

#[test]
fn day1() {
    let input = load_file("1.txt");
    let mut sum = 0;
    for line in input.lines() {
        let mut digits = v![@iter c - b'0', for c in line.as_bytes(), if c.is_ascii_digit()];
        let first = digits.next().unwrap();
        let last = digits.last().unwrap_or(first);
        let n = (first as u32) * 10 + last as u32;
        // println!("{line} = {n}");
        sum += n;
    }

    assert_eq!(sum, 54390);

    // part 2
    let nums = &[
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", //
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ];
    let mut sum = 0;
    for line in input.lines() {
        let (_pos, first) = nums
            .iter()
            .enumerate()
            .filter_map(|(n, s)| line.find(s).map(|i| (i, n)))
            .min()
            .unwrap();
        let (_pos, last) = nums
            .iter()
            .enumerate()
            .filter_map(|(n, s)| line.rfind(s).map(|i| (i, n)))
            .max()
            .unwrap();

        let f = |x| if x > 9 { x - 9 } else { x };
        let n = f(first as u32) * 10 + f(last as u32);
        // println!("{line} = {n}");
        sum += n;
    }

    assert_eq!(sum, 54277);
}

#[test]
fn day2() {
    let input = load_file("2.txt");
    let mut possible = vec![]; // part 1
    let mut sum_power = 0; // part 2
    for line in input.lines() {
        println!("{line}");
        let (game, rounds) = line.split_once(": ").unwrap();
        let game_id = game.split_once(' ').unwrap().1.parse::<u32>().unwrap();
        let mut rgbs = vec![];
        for round in rounds.split("; ") {
            let mut rgb = [0; 3];
            for ball in round.split(", ") {
                let (count, color) = ball.split_once(' ').unwrap();
                let count = count.parse::<u32>().unwrap();
                let i = match color {
                    "red" => 0,
                    "green" => 1,
                    "blue" => 2,
                    _ => panic!("Unknown color {}", color),
                };
                rgb[i] += count;
            }
            println!("{rgb:?}");
            rgbs.push(rgb);
        }

        if rgbs
            .iter()
            .all(|rgb| rgb[0] <= 12 && rgb[1] <= 13 && rgb[2] <= 14)
        {
            possible.push(game_id);
        }

        // part 2, calculate necessary numbers of each color
        let r = rgbs.iter().map(|rgb| rgb[0]).max().unwrap();
        let g = rgbs.iter().map(|rgb| rgb[1]).max().unwrap();
        let b = rgbs.iter().map(|rgb| rgb[2]).max().unwrap();
        let power = r * g * b;
        sum_power += power;
    }

    assert_eq!(possible.iter().sum::<u32>(), 2447);
    assert_eq!(sum_power, 56322);
}

#[test]
fn day3() {
    let input = load_file("3.txt");
    let lines = input.lines().collect::<Vec<_>>();

    let is_symbol = |y: usize, x: usize| -> bool {
        let ch: u8 = lines[y].as_bytes()[x];
        !(ch.is_ascii_digit() || ch == b'.')
    };

    let above_below = |y: usize, x: usize| {
        is_symbol(y, x)
            || (y > 0 && is_symbol(y - 1, x))
            || (y < lines.len() - 1 && is_symbol(y + 1, x))
    };

    #[derive(Clone)]
    struct Part {
        x: usize,
        y: usize,
        len: usize,
        num: u32,
    }

    impl Part {
        fn is_adj(&self, y: usize, x: usize) -> bool {
            (y as i32 - self.y as i32).abs() <= 1
                && ((x as i32 - self.x as i32).abs() <= 1
                    || (x as i32 - (self.x + self.len - 1) as i32).abs() <= 1)
        }
    }

    let mut parts = vec![];
    for (l, line) in lines.iter().enumerate() {
        let mut num = String::new();
        // have we seen a symbol yet?
        let mut adjacent = false;

        for (i, ch) in line.char_indices() {
            adjacent = adjacent || above_below(l, i);
            if ch.is_ascii_digit() {
                num.push(ch);
            } else {
                if adjacent && !num.is_empty() {
                    parts.push(Part {
                        x: i - num.len(),
                        y: l,
                        len: num.len(),
                        num: num.parse::<u32>().unwrap(),
                    });
                }
                num.clear();
                adjacent = above_below(l, i);
            }
        }
        if adjacent && !num.is_empty() {
            parts.push(Part {
                x: line.len() - num.len(),
                y: l,
                len: num.len(),
                num: num.parse::<u32>().unwrap(),
            });
        }
        num.clear();
    }

    assert_eq!(parts.iter().map(|p| p.num).sum::<u32>(), 530495);

    // part 2
    let mut gears = HashMap::<(usize, usize), Vec<Part>>::new();
    for (y, line) in lines.iter().enumerate() {
        for (x, ch) in line.char_indices() {
            if ch == '*' {
                for part in &parts {
                    if part.is_adj(y, x) {
                        gears.entry((y, x)).or_default().push(part.clone());
                    }
                }
            }
        }
    }

    let ratios = Iterator::sum::<u32>(v![@iter
        ps[0].num * ps[1].num,
        for ps in gears.values(), if ps.len() == 2
    ]);

    assert_eq!(ratios, 80253814);
}

#[test]
fn day4() {
    let input = load_file("4.txt");
    let nums_set = |s: &str| num_vec::<usize>(s).into_iter().collect::<HashSet<_>>();

    let n_winners: Vec<usize> = v![{
        let (_card_num, line) = line.split_once(": ").unwrap();
        let (winners, have) = line.split_once(" | ").unwrap();
        nums_set(winners).intersection(&nums_set(have)).count()
    }, for line in input.lines()];

    let mut points = 0;
    for &n in &n_winners {
        if n > 0 {
            points += 1 << (n - 1);
        }
    }
    assert_eq!(points, 28750);

    // part 2
    let mut copies = vec![1; n_winners.len()];
    for i in 0..n_winners.len() {
        for j in 0..n_winners[i] {
            copies[i + j + 1] += copies[i];
        }
    }
    assert_eq!(copies.iter().sum::<u32>(), 10212704);
}

pub fn num_vec<T: FromStr>(s: &str) -> Vec<T> {
    let fail = |_| panic!("Failed to parse {}", s);
    v![n.parse::<T>().unwrap_or_else(fail), for n in s.split_whitespace()]
}

pub fn minmax<T: Ord>(a: T, b: T) -> (T, T) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

#[test]
fn day5() {
    let input = load_file("5.txt");
    let mut lines = input.lines();

    let line = lines.next().unwrap();
    let seeds = num_vec(line.split_once("seeds: ").unwrap().1);
    let empty = lines.next().unwrap();
    assert!(empty.is_empty());

    struct Map {
        ranges: Vec<[i64; 3]>,
    }

    impl Map {
        fn get(&self, i: i64) -> i64 {
            for &[dst, src, len] in &self.ranges {
                if (src..src + len).contains(&i) {
                    return dst + i - src;
                }
            }
            i
        }

        fn split(&self, iv: Ival) -> Vec<Ival> {
            let mut todo = vec![iv];
            let mut done = vec![];
            for &[dst, src, len] in &self.ranges {
                let mut new_todo = vec![];
                for iv in todo {
                    let mapped_start = iv.start.max(src);
                    let mapped_end = iv.end.min(src + len);
                    let dst_mapped = mapped_start + dst - src..mapped_end + dst - src;
                    if dst_mapped.is_empty() {
                        new_todo.push(iv)
                    } else {
                        done.push(dst_mapped);
                        let starts = minmax(iv.start, mapped_start);
                        let ends = minmax(iv.end, mapped_end);
                        new_todo.push(starts.0..starts.1);
                        new_todo.push(ends.0..ends.1);
                    }
                }
                todo = new_todo;
                todo.retain(|iv| !iv.is_empty());
            }
            done.extend(todo);
            done
        }
    }

    let mut done = false;
    let mut maps = vec![];
    while !done {
        let line = lines.next().unwrap();
        let mut map = Map { ranges: vec![] };
        assert!(line.ends_with("map:"));

        loop {
            match lines.next() {
                None => {
                    done = true;
                    break;
                }
                Some("") => break,
                Some(line) => map.ranges.push(num_vec(line).try_into().unwrap()),
            }
        }

        maps.push(map);
    }

    let mut min_dest = i64::MAX;
    for &seed in &seeds {
        let mut i = seed;
        for map in &maps {
            i = map.get(i);
        }
        min_dest = min_dest.min(i);
    }

    assert_eq!(min_dest, 57075758);

    // part 2

    type Ival = Range<i64>;

    let mut min_dest = i64::MAX;
    for x in seeds.chunks(2) {
        let (src, len) = (x[0], x[1]);
        let iv = src..src + len;
        let mut ivs = vec![iv];
        for map in &maps {
            ivs = ivs
                .into_iter()
                .flat_map(|iv| map.split(iv))
                .collect::<Vec<_>>();
        }

        for iv in ivs {
            min_dest = min_dest.min(iv.start);
        }
    }
    assert_eq!(min_dest, 31161857);
}

#[test]
fn day6() {
    let input = load_file("6.txt");
    let lines: Vec<String> = v![l.to_owned(), for l in input.lines()];
    let times = num_vec::<f64>(lines[0].trim_start_matches("Time: "));
    let dists = num_vec::<f64>(lines[1].trim_start_matches("Distance: "));
    assert_eq!(times.len(), dists.len());

    // dist = hold * (duration - hold)
    // is maximized when hold = duration / 2
    // 0 = hold * (duration - hold) - dist
    // 0 = hold * duration - hold * hold - dist
    // 0 = -hold^2 + hold * duration - dist
    //
    // solutions = (-b +- sqrt(b^2 - 4ac)) / 2a

    fn ways_to_win(duration: f64, dist: f64) -> u32 {
        let (a, b, c) = (-1.0, duration, -dist);
        let x = (b * b - 4.0 * a * c).sqrt();
        let sol1 = (-b + x) / (2.0 * a);
        let sol2 = (-b - x) / (2.0 * a);
        sol2.floor() as u32 - sol1.ceil() as u32 + 1
    }

    let mut product = 1;
    for (&duration, &dist) in times.iter().zip(&dists) {
        product *= ways_to_win(duration, dist);
    }

    assert_eq!(product, 160816);

    // part 2
    let mut time_line = lines[0].clone();
    time_line.retain(|c| c.is_ascii_digit());
    let time = time_line.parse::<f64>().unwrap();

    let mut dist_line = lines[1].clone();
    dist_line.retain(|c| c.is_ascii_digit());
    let dist = dist_line.parse::<f64>().unwrap();

    assert_eq!(ways_to_win(time, dist), 46561107);
}

#[test]
fn day7() {
    let input = load_file("7.txt");

    fn do_it(input: &str, cards: &str, sort_counts: impl Fn(&mut [u32])) -> usize {
        let mut hands = vec![];
        for line in input.lines() {
            let (hand, bid) = line.split_once(' ').unwrap();
            let mut decoded_hand = [0; 5];
            let mut counts = [0; 13];
            for (ch, spot) in hand.chars().zip(&mut decoded_hand) {
                let i = cards.find(ch).unwrap();
                *spot = i;
                counts[i] += 1;
            }
            sort_counts(&mut counts);
            hands.push((counts, decoded_hand, bid.parse::<usize>().unwrap()));
        }

        hands.sort();

        let mut total_winnings = 0;
        for (i, (_counts, _hand, bid)) in hands.iter().enumerate() {
            // println!("{:?} {:?} {:?}", counts, hand, bid);
            total_winnings += (i + 1) * bid;
        }
        total_winnings
    }

    let part1 = do_it(&input, "23456789TJQKA", |counts| {
        counts.sort_by(|a, b| b.cmp(a))
    });
    assert_eq!(part1, 250058342);

    let part2 = do_it(&input, "J23456789TQKA", |counts| {
        let jokers = std::mem::take(&mut counts[0]);
        counts.sort_by(|a, b| b.cmp(a));
        counts[0] += jokers;
    });
    assert_eq!(part2, 250506580);
}

pub fn lcm(ns: &[usize]) -> usize {
    let mut ns = ns.to_owned();
    let mut factors = vec![];
    for i in 2.. {
        let mut keep_going = true;
        while keep_going {
            keep_going = false;
            for n in &mut ns {
                if *n % i == 0 {
                    *n /= i;
                    if !keep_going {
                        factors.push(i);
                        keep_going = true;
                    }
                }
            }
        }
        if ns.iter().all(|&n| n == 1) {
            break;
        }
    }

    factors.iter().product::<usize>()
}

#[test]
fn lcm_test() {
    assert_eq!(lcm(&[2, 3, 4]), 12);
    assert_eq!(lcm(&[4]), 4);
    assert_eq!(lcm(&[]), 1);
}

#[test]
fn day8() {
    let input = load_file("8.txt");
    let mut lines = input.lines();
    let directions = lines.next().unwrap();
    assert!(lines.next().unwrap().is_empty());

    type Node = [u8; 3];
    let mut map = HashMap::<Node, (Node, Node)>::new();

    for line in lines {
        // XKM = (FRH, RLM)
        let bytes = line.as_bytes();
        let src = bytes[0..3].try_into().unwrap();
        let left = bytes[7..10].try_into().unwrap();
        let right = bytes[12..15].try_into().unwrap();
        map.insert(src, (left, right));
    }

    let do_step = |node: Node, direction: char| -> Node {
        let (left, right) = map[&node];
        match direction {
            'L' => left,
            'R' => right,
            _ => panic!("Unknown direction {}", direction),
        }
    };

    let trace = |mut location| {
        let mut num_steps = 0;
        loop {
            for step in directions.chars() {
                num_steps += 1;
                location = do_step(location, step);
                if location == [b'Z'; 3] {
                    return num_steps;
                }
            }
        }
    };

    assert_eq!(trace([b'A'; 3]), 22411);

    // part 2

    let starting_nodes: Vec<Node> = v![*n, for n in map.keys(), if n[2] == b'A'];

    struct CycleInfo {
        cycle_start: usize,
        cycle_len: usize,
        success_time: usize,
    }

    impl CycleInfo {
        fn is_done(&self, time: usize) -> bool {
            if time < self.success_time {
                return false;
            }
            let success_cycle_time = self.success_time - self.cycle_start;
            let cycle_time = time - self.cycle_start;
            (cycle_time % self.cycle_len) == success_cycle_time
        }
    }

    let compute_cycle = |start: Node| -> CycleInfo {
        //                       path_i, loc,  total_steps
        let mut seen = HashMap::<(usize, Node), usize>::new();
        let mut location = start;
        let mut path = directions.char_indices().cycle();
        for t in 0.. {
            let (i, step) = path.next().unwrap();
            match seen.entry((i, location)) {
                Entry::Vacant(e) => e.insert(t),
                Entry::Occupied(e) => {
                    let cycle_start = *e.get();
                    let success_times = v![*t, for ((_i, node), t) in &seen, if node[2] == b'Z'];
                    assert_eq!(success_times.len(), 1);
                    return CycleInfo {
                        cycle_start,
                        cycle_len: t - cycle_start,
                        success_time: success_times[0],
                    };
                }
            };
            location = do_step(location, step);
        }
        panic!();
    };

    let trace_until_time = |mut location, stop_time: usize| {
        let mut path = directions.chars().cycle();
        for _ in 0..stop_time {
            location = do_step(location, path.next().unwrap());
        }
        location
    };

    let mut cycles = vec![];
    for &start in &starting_nodes {
        let cycle = compute_cycle(start);
        assert_eq!(trace_until_time(start, cycle.success_time)[2] as char, 'Z');
        assert_eq!(
            trace_until_time(start, cycle.success_time),
            trace_until_time(start, cycle.success_time + cycle.cycle_len),
        );
        assert!(cycle.is_done(cycle.success_time));
        cycles.push(cycle);
    }

    let mut done = vec![false; starting_nodes.len()];
    let mut t = 0;
    while !done.iter().all(|&d| d) {
        let done_cycle_lens = v![c.cycle_len, for (c, d) in cycles.iter().zip(&done), if **d];
        t += lcm(&done_cycle_lens);

        for (cycle, cycle_done) in cycles.iter_mut().zip(&mut done) {
            *cycle_done = cycle.is_done(t);
        }
    }

    for cycle in &cycles {
        assert!(cycle.is_done(t));
    }

    assert_eq!(t, 11188774513823);
}

#[test]
fn day9() {
    let input = load_file("9.txt");
    let mut lines = input.lines();

    fn estimate(nums: &[i64], part: Part) -> i64 {
        if nums.iter().all(|&n| n == 0) {
            return 0;
        }
        let next: Vec<i64> = v![w[1] - w[0], for w in nums.windows(2)];
        match part {
            Part1 => nums[nums.len() - 1] + estimate(&next, part),
            Part2 => nums[0] - estimate(&next, part),
        }
    }

    let (mut part1, mut part2) = (0, 0);
    for line in &mut lines {
        let nums = num_vec::<i64>(line);
        part1 += estimate(&nums, Part1);
        part2 += estimate(&nums, Part2);
    }

    assert_eq!(part1, 1898776583);
    assert_eq!(part2, 1100);
}
