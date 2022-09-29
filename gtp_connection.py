"""
gtp_connection.py
Module for playing games of Go using GoTextProtocol

Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller.
Parts of this code were originally based on the gtp module 
in the Deep-Go project by Isaac Henrion and Amos Storkey 
at the University of Edinburgh.
"""
from logging import exception
from secrets import choice
import traceback
import numpy as np
import re
from random import choice
from sys import stdin, stdout, stderr
from typing import Any, Callable, Dict, List, Tuple

from board_base import (
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    GO_COLOR, GO_POINT,
    PASS,
    MAXSIZE,
    coord_to_point,
    opponent
)
from board import GoBoard
from board_util import GoBoardUtil
from engine import GoEngine

class GtpConnection:
    def __init__(self, go_engine: GoEngine, board: GoBoard, debug_mode: bool = False) -> None:
        """
        Manage a GTP connection for a Go-playing engine

        Parameters
        ----------
        go_engine:
            a program that can reply to a set of GTP commandsbelow
        board: 
            Represents the current board state.
        """
        self._debug_mode: bool = debug_mode
        self.go_engine = go_engine
        self.board: GoBoard = board
        self.commands: Dict[str, Callable[[List[str]], None]] = {
            "protocol_version": self.protocol_version_cmd,
            "quit": self.quit_cmd,
            "name": self.name_cmd,
            "boardsize": self.boardsize_cmd,
            "showboard": self.showboard_cmd,
            "clear_board": self.clear_board_cmd,
            "komi": self.komi_cmd,
            "version": self.version_cmd,
            "known_command": self.known_command_cmd,
            "genmove": self.genmove_cmd,
            "list_commands": self.list_commands_cmd,
            "play": self.play_cmd,
            "legal_moves": self.legal_moves_cmd,
            "gogui-rules_legal_moves": self.gogui_rules_legal_moves_cmd,
            "gogui-rules_final_result": self.gogui_rules_final_result_cmd,
            
            "sidetomove": self.gogui_rules_side_to_move_cmd,
        }

        # argmap is used for argument checking
        # values: (required number of arguments,
        #          error message on argnum failure)
        self.argmap: Dict[str, Tuple[int, str]] = {
            "boardsize": (1, "Usage: boardsize INT"),
            "komi": (1, "Usage: komi FLOAT"),
            "known_command": (1, "Usage: known_command CMD_NAME"),
            "genmove": (1, "Usage: genmove {w,b}"),
            "play": (2, "Usage: play {b,w} MOVE"),
            "legal_moves": (1, "Usage: legal_moves {w,b}"),
        }

    def write(self, data: str) -> None:
        stdout.write(data)

    def flush(self) -> None:
        stdout.flush()

    def start_connection(self) -> None:
        """
        Start a GTP connection. 
        This function continuously monitors standard input for commands.
        """
        line = stdin.readline()
        while line:
            self.get_cmd(line)
            line = stdin.readline()

    def get_cmd(self, command: str) -> None:
        """
        Parse command string and execute it
        """
        if len(command.strip(" \r\t")) == 0:
            return
        if command[0] == "#":
            return
        # Strip leading numbers from regression tests
        if command[0].isdigit():
            command = re.sub("^\d+", "", command).lstrip()

        elements: List[str] = command.split()
        if not elements:
            return
        command_name: str = elements[0]
        args: List[str] = elements[1:]
        if self.has_arg_error(command_name, len(args)):
            return
        if command_name in self.commands:
            try:
                self.commands[command_name](args)
            except Exception as e:
                self.debug_msg("Error executing command {}\n".format(str(e)))
                self.debug_msg("Stack Trace:\n{}\n".format(traceback.format_exc()))
                raise e
        else:
            self.debug_msg("Unknown command: {}\n".format(command_name))
            self.error("Unknown command")
            stdout.flush()

    def has_arg_error(self, cmd: str, argnum: int) -> bool:
        """
        Verify the number of arguments of cmd.
        argnum is the number of parsed arguments
        """
        if cmd in self.argmap and self.argmap[cmd][0] != argnum:
            self.error(self.argmap[cmd][1])
            return True
        return False

    def debug_msg(self, msg: str) -> None:
        """ Write msg to the debug stream """
        if self._debug_mode:
            stderr.write(msg)
            stderr.flush()

    def error(self, error_msg: str) -> None:
        """ Send error msg to stdout """
        stdout.write("? {}\n\n".format(error_msg))
        stdout.flush()

    def respond(self, response: str = "") -> None:
        """ Send response to stdout """
        stdout.write("= {}\n\n".format(response))
        stdout.flush()

    def reset(self, size: int) -> None:
        """
        Reset the board to empty board of given size
        """
        self.board.reset(size)

    def board2d(self) -> str:
        return str(GoBoardUtil.get_twoD_board(self.board))

    def protocol_version_cmd(self, args: List[str]) -> None:
        """ Return the GTP protocol version being used (always 2) """
        self.respond("2")

    def quit_cmd(self, args: List[str]) -> None:
        """ Quit game and exit the GTP interface """
        self.respond()
        exit()

    def name_cmd(self, args: List[str]) -> None:
        """ Return the name of the Go engine """
        self.respond(self.go_engine.name)

    def version_cmd(self, args: List[str]) -> None:
        """ Return the version of the  Go engine """
        self.respond(str(self.go_engine.version))

    def clear_board_cmd(self, args: List[str]) -> None:
        """ clear the board """
        self.reset(self.board.size)
        self.respond()

    def boardsize_cmd(self, args: List[str]) -> None:
        """
        Reset the game with new boardsize args[0]
        """
        self.reset(int(args[0]))
        self.respond()

    def showboard_cmd(self, args: List[str]) -> None:
        self.respond("\n" + self.board2d())

    def komi_cmd(self, args: List[str]) -> None:
        """
        Set the engine's komi to args[0]
        """
        self.go_engine.komi = float(args[0])
        self.respond()

    def known_command_cmd(self, args: List[str]) -> None:
        """
        Check if command args[0] is known to the GTP interface
        """
        if args[0] in self.commands:
            self.respond("true")
        else:
            self.respond("false")

    def list_commands_cmd(self, args: List[str]) -> None:
        """ list all supported GTP commands """
        self.respond(" ".join(list(self.commands.keys())))

    def legal_moves_cmd(self, args: List[str]) -> None:
        """
        List legal moves for color args[0] in {'b','w'}
        """
        board_color: str = args[0].lower()
        color: GO_COLOR = color_to_int(board_color)
        moves: List[GO_POINT] = GoBoardUtil.generate_legal_moves(self.board, color)
        gtp_moves: List[str] = []
        for move in moves:
            coords: Tuple[int, int] = point_to_coord(move, self.board.size)
            gtp_moves.append(format_point(coords))
        sorted_moves = " ".join(sorted(gtp_moves))
        self.respond(sorted_moves)

    """
    ==========================================================================
    Assignment 1 - game-specific commands start here
    ==========================================================================
    """
    """
    ==========================================================================
    Assignment 1 - commands we already implemented for you
    ==========================================================================
    """
    def gogui_analyze_cmd(self, args):
        """ We already implemented this function for Assignment 1 """
        self.respond("pstring/Legal Moves For ToPlay/gogui-rules_legal_moves\n"
                     "pstring/Side to Play/gogui-rules_side_to_move\n"
                     "pstring/Final Result/gogui-rules_final_result\n"
                     "pstring/Board Size/gogui-rules_board_size\n"
                     "pstring/Rules GameID/gogui-rules_game_id\n"
                     "pstring/Show Board/gogui-rules_board\n"
                     )

    def gogui_rules_game_id_cmd(self, args):
        """ We already implemented this function for Assignment 1 """
        self.respond("NoGo")

    def gogui_rules_board_size_cmd(self, args):
        """ We already implemented this function for Assignment 1 """
        self.respond(str(self.board.size))

    def gogui_rules_side_to_move_cmd(self, args):
        """ We already implemented this function for Assignment 1 """
        color = "black" if self.board.current_player == BLACK else "white"
        self.respond(color)

    def gogui_rules_board_cmd(self, args):
        """ We already implemented this function for Assignment 1 """
        size = self.board.size
        str = ''
        for row in range(size-1, -1, -1):
            start = self.board.row_start(row + 1)
            for i in range(size):
                #str += '.'
                point = self.board.board[start + i]
                if point == BLACK:
                    str += 'X'
                elif point == WHITE:
                    str += 'O'
                elif point == EMPTY:
                    str += '.'
                else:
                    assert False
            str += '\n'
        self.respond(str)

    """
    ==========================================================================
    Assignment 1 - game-specific commands you have to implement or modify
    ==========================================================================
    """
    def gogui_rules_final_result_cmd(self, args):
        """ Implement this function for Assignment 1 """
        #checks for winning conditiong of the game
        #checks which player is playing the last move
        #responds the winner player 

        legal = self.legal_moves_check(self.board.current_player)
        if len(legal) == 0:
            if self.board.current_player == 1:
                self.respond('white')
            
            else:
                self.respond('black')

        else:
            self.respond('unknown')
        return
         

            
        


    def legal_moves_check(self, board_move):
        #checks for all the legal moves on the current board
        #appends all the legal moves in a list
        #returns the sorted list of legal moves 

        legal_moves_list = []
        empty_points = self.board.get_empty_points()
        for point in empty_points:
            if not self.captureErr(point, board_move):
                if self.board.is_legal(point, board_move):

                    legal_moves_list.append(format_point(point_to_coord(point, self.board.size)))
                    
        legal_moves_list.sort(key=lambda point: point[0])
        sorted_moves = " ".join(legal_moves_list)
                        
        return sorted_moves

    def gogui_rules_legal_moves_cmd(self, args):
        """ Implement this function for Assignment 1 """
        #responds the legal moves list 

        legal_moves_list = self.legal_moves_check(self.board.current_player)
        self.respond(legal_moves_list)
        return
        
   
    def occupiedErr(self, board_move):
        #this method checks if the move is on empty points
        #it returns false if move is empty and returns true if not

        if board_move in self.board.get_empty_points():
            return False
        else:
            return True

    def captureErr(self, move, color):
        #this method checks if moves in a capture 
        #if it is in capture then retuens true or else false

        copyBoard = self.board.copy()
        if color == 2:
            oppositecolor = 1
        else:
            oppositecolor = 2
        Neighborbeforemove = copyBoard.neighbors_of_color(move, oppositecolor)
        copyBoard.play_move(move, color)
        Neighboraftermove = copyBoard.neighbors_of_color(move, oppositecolor)
        if len(Neighborbeforemove) != len(Neighboraftermove):
            return True
        else:
            return False

    def play_cmd(self, args: List[str]) -> None:
        """
        play a move args[1] for given color args[0] in {'b','w'}
        """
        try:
            
            board_color = args[0].lower()
            #checks for valid color input
            if board_color == 'b'  or board_color == 'w':
                board_color_int = color_to_int(board_color)
            else:
                self.respond('illegal move: "{} {}" wrong color'.format(args[0], args[1]))
                return  
            try:
                #checks for wrong coordinates
                if str(args[1]).lower() == "pass":
                    #self.board.play_move(PASS, color)
                    #self.board.current_player = opponent(color)
                    self.respond('illegal move: "{} {}" wrong coordinate'.format(args[0],args[1]))
                    return
                coord = move_to_coord(args[1], self.board.size)
            except Exception as a:
                self.respond('illegal move: "{} {}" wrong coordinate'.format(args[0],args[1]))
                return

            board_move = coord_to_point(coord[0], coord[1], self.board.size)
            #checks for occupied points
            if self.occupiedErr(board_move):
                self.respond('illegal move: "{} {}" occupied'.format(args[0],args[1]))
                return
            #checks for capture
            if self.captureErr(board_move, board_color_int):
                self.respond('illegal move: "{} {}" capture'.format(args[0],args[1]))
                return
            #checks for illegal move
            if not self.board.play_move(board_move, board_color_int):
                self.respond("Illegal Move: {}".format(args[0], args[1]))
                return
            else:
                self.debug_msg(
                    "Move: {}\nBoard:\n{}\n".format(board_move, self.board2d())
                )
            self.respond()
        except Exception as e:
            self.respond("Error: {}".format(str(e)))

    def genmove_cmd(self, args: List[str]) -> None:
        """ generate a move for color args[0] in {'b','w'} """
        board_color = args[0].lower()
        board_color_int = color_to_int(board_color)
        emptypoints = self.board.get_empty_points()
        legal = []
        for point in emptypoints:
            if not self.captureErr(point, board_color_int):
                if self.board.is_legal(point,board_color_int):
                    legal.append(point)
        #if there are legal position left in board then play move randomly
        if len(legal) > 0:
            move = choice(legal)
            #move = self.go_engine.get_move(self.board, color)
            move_coord = point_to_coord(move, self.board.size)
            move_as_string = format_point(move_coord)
        #if self.board.is_legal(move, color):
            self.board.play_move(move, self.board.current_player)
            self.respond(move_as_string)
            return
        else:
            #self.respond("Illegal move: {}".format(move_as_string))
            self.respond("resign")

    """
    ==========================================================================
    Assignment 1 - game-specific commands end here
    ==========================================================================
    """

def point_to_coord(point: GO_POINT, boardsize: int) -> Tuple[int, int]:
    """
    Transform point given as board array index 
    to (row, col) coordinate representation.
    Special case: PASS is not transformed 
    """
    if point == PASS:
        return PASS
    else:
        NS = boardsize + 1
        return divmod(point, NS)


def format_point(move: Tuple[int, int]) -> str:
    """
    Return move coordinates as a string such as 'A1', or 'PASS'.
    """
    assert MAXSIZE <= 25
    column_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    if move[0] == PASS:
        return "PASS"
    row, col = move
    if not 0 <= row < MAXSIZE or not 0 <= col < MAXSIZE:
        raise ValueError
    return column_letters[col - 1] + str(row)


def move_to_coord(point_str: str, board_size: int) -> Tuple[int, int]:
    """
    Convert a string point_str representing a point, as specified by GTP,
    to a pair of coordinates (row, col) in range 1 .. board_size.
    Raises ValueError if point_str is invalid
    """
    if not 2 <= board_size <= MAXSIZE:
        raise ValueError("board_size out of range")
    s = point_str.lower()
    if s == "pass":
        return PASS
    try:
        col_c = s[0]
        if (not "a" <= col_c <= "z") or col_c == "i":
            raise ValueError
        col = ord(col_c) - ord("a")
        if col_c < "i":
            col += 1
        row = int(s[1:])
        if row < 1:
            raise ValueError
    except (IndexError, ValueError):
        raise ValueError("invalid point: '{}'".format(s))
    if not (col <= board_size and row <= board_size):
        raise ValueError("point off board: '{}'".format(s))
    return row, col


def color_to_int(c: str) -> int:
    """convert character to the appropriate integer code"""
    color_to_int = {"b": BLACK, "w": WHITE, "e": EMPTY, "BORDER": BORDER}
    return color_to_int[c]
