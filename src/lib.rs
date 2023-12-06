use std::{collections::HashMap, ops::Range, str::FromStr};

pub fn load_file(filename: &str) -> String {
    std::fs::read_to_string(format!("inputs/{filename}"))
        .unwrap_or_else(|_err| panic!("Failed to read {}", filename))
}

#[test]
fn day1() {
    let input = load_file("1.txt");
    let mut sum = 0;
    for line in input.lines() {
        let mut digits = line
            .as_bytes()
            .iter()
            .filter(|c| c.is_ascii_digit())
            .map(|c| c - b'0'); // convert from ascii
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

    let ratios = gears
        .values()
        .filter(|ps| ps.len() == 2)
        .map(|ps| ps[0].num * ps[1].num)
        .sum::<u32>();

    assert_eq!(ratios, 80253814);
}

#[test]
fn day4() {
    let input = load_file("4.txt");
    let nums_set = |s: &str| {
        s.split_whitespace()
            .map(|n| n.trim().parse::<u32>().unwrap())
            .collect::<std::collections::HashSet<_>>()
    };

    let n_winners: Vec<usize> = input
        .lines()
        .map(|line| {
            let (_card_num, line) = line.split_once(": ").unwrap();
            let (winners, have) = line.split_once(" | ").unwrap();
            nums_set(winners).intersection(&nums_set(have)).count()
        })
        .collect();

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

pub fn num_vec<T>(s: &str) -> Vec<T>
where
    T: FromStr,
    T::Err: std::fmt::Debug,
{
    s.split_whitespace()
        .map(|n| n.trim().parse::<T>().unwrap())
        .collect()
}

fn minmax<T: Ord>(a: T, b: T) -> (T, T) {
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
